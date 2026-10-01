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
    t[:3, 3] = [base[0] + .12 * np.cos(base[2]), base[1] + .12 * np.sin(base[2]), mount_z]
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
        self.action_space=action_space;self.observation_space=observation_space
    def robot(self,s):
        o=s.get_object_from_name('robot');return np.array([s.get(o,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]])
    def xyz(self,s,o):return np.array([s.get(o,f) for f in ['x','y','z']])
    def reset(self,s,info):
        self.stage=-1;self.ticks=0;self.steps=0;self.qt=None;self.base_target=None
        typ=self.observation_space.get_type('mujoco_movable_object')
        self.cubes=[o for o in s.get_objects(typ) if o.name.startswith('cube_')]
        self.source=s.get_object_from_name('bin_yellow_0');self.green=s.get_object_from_name('bin_green_0')
        self.src=self.xyz(s,self.source);self.dst=self.xyz(s,self.green);self.rim=self.src+[-.22,0,0]
    def get_action(self,s):
        self.ticks+=1;self.steps+=1;r=self.robot(s);o=s.get_object_from_name('robot')
        v=np.array([s.get(o,'vel_arm_joint'+str(i)) for i in range(1,8)])
        a=np.zeros(11,dtype=np.float32);a[-1]=float(self.stage>=2)
        if self.stage==-1:
            delta=np.array([-.2093,.041,.0563])-r[:3]
            a[:3]=np.clip(delta,-.06,.06)
            if self.ticks>=12:
                self.stage=0;self.ticks=0
            return a
        if self.stage>=7:
            if self.qt is None:
                cp=np.mean([self.xyz(s,obj) for obj in self.cubes],axis=0) if self.cubes else self.xyz(s,self.source)
                self.base_target=r[:3].copy()
                ee=fk(r[3:],r[:3]);target=ee[:3,3].copy();target[:2]+=self.xyz(s,self.green)[:2]-cp[:2]
                self.qt,_=ik(target,ee[:3,:3],r[3:],r[:3])
            a[3:10]=np.clip(2.5*(self.qt-r[3:])+.15*v,-.1,.1)
            if self.stage==7:
                a[:3]=np.clip(self.base_target-r[:3],-.025,.025)
                if self.ticks>=50:self.stage=8;self.ticks=0
            else:a[-1]=0
            return a
        down=np.diag([1.,-1.,-1.]);angle=0
        if self.stage==4:
            if self.base_target is None:
                self.base_target=r[:3].copy();self.base_target[1]+=self.dst[1]-self.src[1]
                self.qt=r[3:].copy()
            a[:3]=np.clip(self.base_target-r[:3],-.045,.045)
            done=max(abs(self.base_target-r[:3]))<.002 and self.ticks>10
            duration=30
        else:
            if self.stage==0:target=self.rim+[0,0,.20];duration=75
            elif self.stage==1:target=self.rim+[0,0,.032];duration=50
            elif self.stage==2:target=self.xyz(s,self.source)+[-.22,0,.032];duration=20
            elif self.stage==3:target=self.rim+[0,0,.39];duration=65
            elif self.stage==5:target=self.rim+[.12,self.dst[1]-self.src[1],.29];duration=85;angle=-1.4
            elif self.stage==6:target=self.rim+[.12,self.dst[1]-self.src[1],.29];duration=75;angle=-1.9
            else:
                target=self.rim+[.12,self.dst[1]-self.src[1],.29];duration=75;angle=-1.9+.12*np.sin(self.ticks*.2)
            if self.qt is None:
                self.qt,_=ik(target,Rotation.from_euler('y',angle).as_matrix()@down,r[3:],r[:3])
            done=False
        a[3:10]=np.clip(2.5*(self.qt-r[3:])+.15*v,-.1,.1)
        if done or self.ticks>=duration:
            self.stage+=1;self.ticks=0;self.qt=None
        return a
