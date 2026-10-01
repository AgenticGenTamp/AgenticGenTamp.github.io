import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from kinematics import fk

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space=action_space
        self.observation_space=observation_space
    def reset(self,state,info):
        self.robot=state.get_object_from_name('robot');self.t=0
        self.cubes=sorted([o for o in state.get_objects(self.observation_space.get_type('mujoco_movable_object')) if o.name.startswith('cube')],key=lambda o:int(o.name[4:]))
        self.stage=0;self.k=0;self.age=0;self.qcache={}
        self.make_plan(state)
    def xyz(self,state,o):return np.array([state.get(o,f) for f in ['x','y','z']])
    def ik(self,z):
        if z not in self.qcache:
            def err(q):
                p,r=fk(q)
                return np.r_[(p+r@np.array([0,0,.12])-np.array([.55,0,z]))*3,Rotation.from_matrix(r.T@np.diag([1,-1,-1])).as_rotvec()]
            self.qcache[z]=least_squares(err,np.array([-.04,.7,3.22,-1.4,-.1,-1.1,.08]),max_nfev=100).x
        return self.qcache[z]
    def make_plan(self,state):
        if not self.cubes:self.plan=[];return
        obj=self.cubes[self.k%len(self.cubes)];pos=self.xyz(state,obj)
        idx=int(obj.name[4:])-1
        bn=['red','green','blue','yellow'][idx%4]
        dest=self.xyz(state,state.get_object_from_name('bin_'+bn))
        srcbase=np.r_[pos[:2]+[.67,0],np.pi];dstbase=np.r_[dest[:2]+[.67,0],np.pi]
        self.plan=[(srcbase,.20,0,75 if self.k==0 else 12),(srcbase,.03,0,26),(srcbase,.03,1,4),(srcbase,.22,1,26),(dstbase,.22,1,10),(dstbase,.22,0,8)]
        self.stage=0;self.age=0
    def get_action(self,state):
        self.t+=1;r=self.robot
        if not self.plan:return np.zeros(11,dtype=np.float32)
        base,z,grip,duration=self.plan[self.stage]
        q=np.array([state.get(r,'pos_arm_joint'+str(i)) for i in range(1,8)])
        v=np.array([state.get(r,'vel_arm_joint'+str(i)) for i in range(1,8)])
        curbase=np.array([state.get(r,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
        a=np.r_[(base-curbase)/np.array([.87,.87,.994]),2.5*(self.ik(z)-q)+.15*v,grip]
        self.age+=1
        if self.age>=duration:
            self.stage+=1;self.age=0
            if self.stage>=len(self.plan):self.k+=1;self.make_plan(state)
        return np.clip(a,self.action_space.low,self.action_space.high).astype(np.float32)
