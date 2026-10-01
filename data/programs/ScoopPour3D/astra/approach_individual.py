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
        o=s.get_object_from_name('robot')
        return np.array([s.get(o,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint'+str(i) for i in range(1,8)]])
    def xyz(self,s,o):return np.array([s.get(o,f) for f in ['x','y','z']])
    def reset(self,s,info):
        self.cubes=[o for o in s.get_objects(self.observation_space.get_type('mujoco_movable_object')) if o.name.startswith('cube_')]
        self.green=s.get_object_from_name('bin_green_0');self.stage='select';self.ticks=0;self.steps=0;self.qt=None;self.tries={};self.base_target=None;self.target=None
    def advance(self,stage):self.stage=stage;self.ticks=0;self.qt=None
    def get_action(self,s):
        self.steps+=1;self.ticks+=1;r=self.robot(s);ee=fk(r[3:],r[:3]);dest=self.xyz(s,self.green)
        a=np.zeros(11,dtype=np.float32)
        ro=s.get_object_from_name('robot');v=np.array([s.get(ro,'vel_arm_joint'+str(i)) for i in range(1,8)])
        if self.stage=='select':
            rem=[o for o in self.cubes if abs(self.xyz(s,o)[0]-dest[0])>.19 or abs(self.xyz(s,o)[1]-dest[1])>.12 or self.xyz(s,o)[2]>.54]
            if not rem:return a
            self.target=min(rem,key=lambda o:self.tries.get(o.name,0)*.15+np.linalg.norm(self.xyz(s,o)[:2]-ee[:2,3]))
            self.pick=self.xyz(s,self.target);self.tries[self.target.name]=self.tries.get(self.target.name,0)+1
            self.base_target=r[:3].copy();self.base_target[0]=min(r[0],-.24);self.base_target[2]=0
            self.advance('above')
        grip=1 if self.stage in ['close','lift','carry','drop'] else 0
        if self.stage in ['above','down','close','lift']:
            if self.stage=='above':pos=self.pick.copy();pos[2]=.61
            elif self.stage in ['down','close']:pos=self.pick.copy();pos[2]+=.008
            else:pos=self.pick.copy();pos[2]=.61
            if self.qt is None:
                self.qt,_=ik(pos,np.diag([1.,-1.,-1.]),r[3:],r[:3])
            d=self.qt-r[3:];a[3:10]=np.clip(2.5*d+.15*v,-.1,.1)
            done=max(abs(d))<.014 and self.ticks>=3
            if self.stage=='above' and (done or self.ticks>105):
                self.pick=self.xyz(s,self.target);self.advance('down')
            elif self.stage=='down' and (done or self.ticks>35):self.advance('close')
            elif self.stage=='close' and self.ticks>=5:self.advance('lift')
            elif self.stage=='lift' and (done or self.ticks>35):
                held=[o for o in self.cubes if self.xyz(s,o)[2]>.54 and np.linalg.norm(self.xyz(s,o)[:2]-ee[:2,3])<.13]
                if held:
                    center=np.mean([self.xyz(s,o) for o in held],axis=0)
                    self.returnbase=r[:3].copy();self.base_target=r[:3].copy();self.base_target[:2]+=dest[:2]-center[:2]
                    self.holdq=r[3:].copy();self.advance('carry')
                else:self.advance('select')
        elif self.stage=='carry':
            d=self.base_target-r[:3];a[:3]=np.clip(d,-.08,.08);a[3:10]=np.clip(2.5*(self.holdq-r[3:])+.15*v,-.1,.1)
            if max(abs(d))<.003 or self.ticks>25:self.advance('drop')
        elif self.stage in ['drop','retract']:
            pos=ee[:3,3].copy();pos[2]=.515 if self.stage=='drop' else .61
            if self.qt is None:self.qt,_=ik(pos,np.diag([1.,-1.,-1.]),r[3:],r[:3])
            d=self.qt-r[3:];a[3:10]=np.clip(2.5*d+.15*v,-.1,.1)
            if (max(abs(d))<.012 and self.ticks>3) or self.ticks>30:
                self.holdq=r[3:].copy();self.advance('release' if self.stage=='drop' else 'return')
        elif self.stage=='return':
            d=self.returnbase-r[:3];a[:3]=np.clip(d,-.06,.06);a[3:10]=np.clip(2.5*(self.holdq-r[3:])+.15*v,-.1,.1)
            if max(abs(d))<.003 or self.ticks>25:self.advance('select')
        elif self.stage=='release':
            a[3:10]=np.clip(2.5*(self.holdq-r[3:])+.15*v,-.1,.1)
            if self.ticks>=8:self.advance('retract')
        a[-1]=grip
        return a
