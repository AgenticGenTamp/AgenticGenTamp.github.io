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
        self.action_space=action_space; self.observation_space=observation_space
    def robot(self,s):
        o=s.get_object_from_name('robot')
        return np.array([s.get(o,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]])
    def transform(self,s,name):
        o=s.get_object_from_name(name);t=np.eye(4)
        t[:3,3]=[s.get(o,f) for f in ['x','y','z']]
        t[:3,:3]=Rotation.from_quat([s.get(o,f) for f in ['qx','qy','qz','qw']]).as_matrix()
        return t
    def reset(self,s,info):
        self.stage=0;self.ticks=0;self.steps=0;self.qtarget=None;self.rel=None;self.point=None
        self.cubes=[o for o in s.get_objects(self.observation_space.get_type('mujoco_movable_object')) if o.name.startswith('cube_')]
        self.heights=[.7,.6,.56,.52,.48]
        self.waypoints=[]
        for h in self.heights:self.waypoints.extend([(h,0,90),(h,1,5),(h+.07,1,35)])
        self.source=self.transform(s,'bin_yellow_0')[:3,3]
        self.dest=self.transform(s,'bin_green_0')[:3,3]
    def get_action(self,s):
        self.steps+=1;self.ticks+=1;rob=self.robot(s);ee=fk(rob[3:],rob[:3]);sp=self.transform(s,'scoop_0')
        if self.stage<15:
            h,g,duration=self.waypoints[self.stage]
            if self.qtarget is None:
                if self.stage%3==0:self.point=sp[:3,3].copy()
                target=self.point.copy();target[2]=h
                self.qtarget,_=ik(target,np.diag([1.,-1.,-1.]),rob[3:],rob[:3])
            diff=self.qtarget-rob[3:]
            arm=diff*min(3.,.1/max(1e-8,max(abs(diff))))
        else:
            g=1
            if self.rel is None:self.rel=np.linalg.inv(ee)@sp
            phase=(self.stage-15)%8
            x,y,z=self.source;dx,dy,dz=self.dest
            targets=[([x+.13,y,.62],0,65),([x+.13,y,z+.04],0,40),([x-.09,y,z+.04],0,60),([x-.09,y,.65],0,40),([dx,dy,.65],0,60),([dx,dy,.61],-1.5,65),([dx,dy,.61],-1.5,15),([dx,dy,.66],0,50)]
            pt,ang,duration=targets[phase]
            if self.qtarget is None:
                st=np.eye(4);st[:3,3]=pt;st[:3,:3]=Rotation.from_euler('y',ang).as_matrix()
                target=st@np.linalg.inv(self.rel)
                self.qtarget,_=ik(target[:3,3],target[:3,:3],rob[3:],rob[:3])
            diff=self.qtarget-rob[3:]
            o=s.get_object_from_name('robot');v=np.array([s.get(o,'vel_arm_joint'+str(i)) for i in range(1,8)])
            arm=np.clip(2.5*diff+.15*v,-.1,.1)
        a=np.zeros(11,dtype=np.float32);a[3:10]=arm;a[-1]=g
        if self.ticks>=duration:
            self.stage+=1;self.ticks=0;self.qtarget=None
        return a
