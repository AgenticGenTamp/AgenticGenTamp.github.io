import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from kinematics import fk

class GeneratedApproach:
    def __init__(self,action_space,observation_space,primitives):
        self.action_space=action_space;self.observation_space=observation_space
        self.qcache={}
    def reset(self,state,info):
        self.robot=state.get_object_from_name('robot');self.t=0;self.stage=0;self.age=0;self.idle=0;self.strict=False;self.angle=0.
        self.cubes=sorted([o for o in state.get_objects(self.observation_space.get_type('mujoco_movable_object')) if o.name.startswith('cube')],key=lambda o:int(o.name[4:]))
        start=min((int(o.name[4:]) for o in self.cubes),default=1)
        self.bins={o.name:state.get_object_from_name('bin_'+['red','green','blue','yellow'][(int(o.name[4:])-start)%4]) for o in self.cubes}
        self.targets={o.name:self.xyz(state,self.bins[o.name]) for o in self.cubes}
        self.attempts={o.name:0 for o in self.cubes};self.obj=None;self.reach=.67;self.armreach=.55
        self.choose(state)
    def xyz(self,state,o):return np.array([state.get(o,f) for f in ['x','y','z']])
    def ik(self,z):
        z=round(float(z),3)
        key=(self.armreach,z)
        if key not in self.qcache:
            def err(q):
                p,r=fk(q)
                return np.r_[(p+r@np.array([0,0,.12])-np.array([self.armreach,0,z]))*3,Rotation.from_matrix(r.T@np.diag([1,-1,-1])).as_rotvec()]
            self.qcache[key]=least_squares(err,np.array([-.04,.7,3.22,-1.4,-.1,-1.1,.08]),max_nfev=70).x
        result=self.qcache[key].copy()
        result[-1]+=self.angle
        return result
    def done(self,state,o):
        p=self.xyz(state,o);d=self.targets[o.name]
        if self.strict:return np.linalg.norm(p[:2]-d[:2])<.035 and abs(p[2]-(d[2]+.02))<.025
        return np.max(np.abs(p[:2]-d[:2]))<.048 and abs(p[2]-(d[2]+.02))<.04
    def choose(self,state):
        todo=[o for o in self.cubes if not self.done(state,o)]
        if not todo:self.obj=None;return
        self.obj=min(todo,key=lambda o:(self.attempts[o.name],-round(self.xyz(state,o)[2],2),-self.xyz(state,o)[0]))
        self.attempts[self.obj.name]+=1
        if len(self.cubes)<=4 and self.attempts[self.obj.name]>=3:self.angle=np.pi/2
        pos=self.xyz(state,self.obj)
        self.armreach=.55 if pos[0]>-.065 and len(self.cubes)<=4 else .65
        self.reach=self.armreach+.12
        self.src=np.r_[pos[:2]+[self.reach,0],np.pi]
        self.pickz=np.clip(pos[2]-.38,.025,.22)
        self.highz=max(.16,self.pickz+.10)
        self.base=self.src.copy();self.z=self.highz;self.grip=0;self.stage=0;self.age=0
    def next_stage(self,state):
        self.stage+=1;self.age=0
        if self.stage==1:
            pos=self.xyz(state,self.obj);self.src=np.r_[pos[:2]+[self.reach,0],np.pi]
            self.base=self.src.copy();self.z=self.pickz
        elif self.stage==2:self.grip=1
        elif self.stage==3:self.z=self.highz
        elif self.stage==4:
            pos=self.xyz(state,self.obj)
            extra=[o for o in self.cubes if o!=self.obj and self.xyz(state,o)[2]>.48 and np.linalg.norm(self.targets[o.name][:2]-self.targets[self.obj.name][:2])>.06]
            if len(self.cubes)<=4 and extra:
                self.grip=0;self.stage=6;self.angle=np.pi/2 if self.angle==0 else 0.;return
            if pos[2]<.475:
                self.grip=0;self.choose(state);return
            rb=self.robot;curbase=np.array([state.get(rb,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
            self.base=curbase.copy();self.base[:2]+=self.targets[self.obj.name][:2]-pos[:2]
        elif self.stage==5:self.grip=0
        else:self.choose(state)
    def get_action(self,state):
        self.t+=1
        self.targets={o.name:self.xyz(state,self.bins[o.name]) for o in self.cubes}
        if self.obj is None:
            self.idle+=1
            if self.idle>8:self.strict=True
            self.choose(state)
            if self.obj is None:return np.zeros(11,dtype=np.float32)
        r=self.robot
        q=np.array([state.get(r,'pos_arm_joint'+str(i)) for i in range(1,8)])
        v=np.array([state.get(r,'vel_arm_joint'+str(i)) for i in range(1,8)])
        curbase=np.array([state.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
        targetq=self.ik(self.z)
        settled=np.max(np.abs(targetq-q))<.018 and np.max(np.abs(self.base-curbase))<.009
        limits=[85,35,3,35,12,5,6]
        self.age+=1
        if self.age>=limits[self.stage] or (self.stage not in (2,5,6) and self.age>=3 and settled):
            self.next_stage(state)
            if self.obj is None:return np.zeros(11,dtype=np.float32)
            targetq=self.ik(self.z)
        a=np.r_[(self.base-curbase)/np.array([.87,.87,.994]),2.5*(targetq-q)+.15*v,self.grip]
        return np.clip(a,self.action_space.low,self.action_space.high).astype(np.float32)
