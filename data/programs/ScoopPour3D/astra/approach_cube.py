import numpy as np
"""Candidate Kinova Gen3 kinematics from public URDF dimensions.

Mount height and gripper extension are configurable because simulation mounting
and gripper geometry have not yet been empirically calibrated.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

_ORIGINS = [((0, 0, .15643), np.pi),
            ((0, .005375, -.12838), np.pi / 2),
            ((0, -.21038, -.006375), -np.pi / 2),
            ((0, .006375, -.21038), np.pi / 2),
            ((0, -.20843, -.006375), -np.pi / 2),
            ((0, 0, -.10593), np.pi / 2),
            ((0, -.10593, 0), -np.pi / 2)]
_ORIGIN_MATS = []
for xyz, rx in _ORIGINS:
    t = np.eye(4)
    t[:3, 3] = xyz
    t[:3, :3] = Rotation.from_euler('x', rx).as_matrix()
    _ORIGIN_MATS.append(t)

def fk(q, base=(0., 0., 0.), mount_z=.4, extension=.12):
    """Return world transform at gripper point; extension along tool +z."""
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('z', base[2]).as_matrix()
    t[:3, 3] = [base[0] + .17 * np.cos(base[2]), base[1] + .17 * np.sin(base[2]), mount_z]
    for o, v in zip(_ORIGIN_MATS, q):
        r = np.eye(4)
        c, s = np.cos(v), np.sin(v)
        r[:2, :2] = [[c, -s], [s, c]]
        t = t @ o @ r
    o = np.eye(4)
    o[:3, 3] = [0, 0, -.061525 - extension]
    o[:3, :3] = Rotation.from_euler('x', np.pi).as_matrix()
    return t @ o

def ik(target_xyz, target_rotation, q0, base=(0., 0., 0.), mount_z=.4, extension=.12, max_nfev=80):
    target_xyz = np.asarray(target_xyz)
    q0 = np.asarray(q0)
    def error(q):
        t = fk(q, base, mount_z, extension)
        pos = (t[:3, 3] - target_xyz)
        rot = Rotation.from_matrix(target_rotation @ t[:3, :3].T).as_rotvec()
        return np.r_[pos, .15 * rot, .002 * (q - q0)]
    opt = least_squares(error, q0, max_nfev=max_nfev, ftol=1e-5, xtol=1e-5)
    return opt.x, np.linalg.norm(error(opt.x)[:3])


class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space = action_space
        self.observation_space = observation_space
    def robot(self, state):
        o=state.get_object_from_name('robot')
        return np.array([state.get(o,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]])
    def reset(self, state, info):
        self.steps=0; self.stage=0; self.ticks=0; self.qtarget=None
        self.cubes=[o for o in state.get_objects(self.observation_space.get_type('mujoco_movable_object')) if o.name.startswith('cube_')]
        obj=min(self.cubes,key=lambda o: abs(state.get(o,'x')-.5)+abs(state.get(o,'y')+.2)) if self.cubes else None
        self.target=np.array([state.get(obj,f) for f in ['x','y','z']]) if obj else np.array([.5,-.2,.48])
        p=self.target
        self.waypoints=[([p[0],p[1],.65],0),([p[0],p[1],.47],0),([p[0],p[1],.47],1),([p[0],p[1],.65],1),([.5,.2,.65],1),([.5,.2,.52],1),([.5,.2,.52],0)]
    def get_action(self, state):
        self.steps+=1; self.ticks+=1
        rob=self.robot(state)
        pos,grip=self.waypoints[min(self.stage,len(self.waypoints)-1)]
        if self.qtarget is None:
            self.qtarget,_=ik(pos,np.diag([1.,-1.,-1.]),rob[3:],rob[:3])
        diff=self.qtarget-rob[3:]
        a=np.zeros(11,dtype=np.float32); vel=np.array([state.get(state.get_object_from_name('robot'),'vel_arm_joint'+str(i)) for i in range(1,8)]); a[3:10]=np.clip(2.5*diff+.15*vel,-.1,.1); a[10]=grip
        if (max(abs(diff))<.035 and self.ticks>8) or self.ticks>[130,60,8,60,100,50,8][min(self.stage,6)]:
            self.stage+=1; self.qtarget=None; self.ticks=0
        return a
