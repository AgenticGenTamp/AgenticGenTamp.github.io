import numpy as np
from scipy.optimize import least_squares
from kinematics import fk, HOME

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.action_space=action_space
        self.os=observation_space
    def pose(self,z,x=.5,tilt=0):
        def residual(v):
            q=HOME.copy();q[[1,3,5]]=v
            p,R=fk(q)
            return np.r_[(p-np.array([x,.001,z]))[[0,2]],R[0,2]-np.sin(tilt),R[2,2]-np.cos(tilt)]
        solution=least_squares(residual,HOME[[1,3,5]],bounds=([-2.24,-2.58,-2.1],[2.24,2.58,2.1]),max_nfev=100).x
        q=HOME.copy();q[[1,3,5]]=solution
        return q
    def reset(self,state,info):
        self.robot=state.get_objects(self.os.get_type('mujoco_tidybot_robot'))[0]
        self.objects=sorted(state.get_objects(self.os.get_type('mujoco_movable_object')),key=lambda o:o.name)
        self.fixture=state.get_objects(self.os.get_type('mujoco_fixture'))[0]
        self.steps=0;self.phase='travel';self.age=0;self.index=0
        self.down=self.pose(-.04,.35);self.up=self.pose(state.get(self.fixture,'z')+.535,.65,-np.pi/2)
        self.target=HOME.copy()
        self.choose(state)
    def choose(self,state):
        # Recheck placements so a displaced object can be retried.
        fx=state.get(self.fixture,'x'); fy=state.get(self.fixture,'y'); fz=state.get(self.fixture,'z')
        remaining=[]
        for offset in range(len(self.objects)):
            idx=(self.index+offset)%len(self.objects); obj=self.objects[idx]
            placed=(abs(state.get(obj,'x')-fx)<.18 and abs(state.get(obj,'y')-fy)<.28 and abs(state.get(obj,'z')-fz-.5667)<.045)
            if not placed:remaining.append(idx)
        if not remaining:
            self.phase='done'
            return
        self.index=remaining[0]
        self.obj=self.objects[self.index]
        self.base=np.array([state.get(self.obj,'x')-.475,state.get(self.obj,'y')-.001,0.])
        n=len(self.objects); rows=max(1,int(np.ceil(n/5))); cols=int(np.ceil(n/rows))
        row=self.index//cols; col=self.index%cols
        targetx=state.get(self.fixture,'x')+.06-.09*row
        targety=state.get(self.fixture,'y')+.10*(col-(cols-1)/2)
        self.dropbase=np.array([targetx-.815,targety-.001,0.])
        self.staging=self.dropbase.copy();self.staging[0]=state.get(self.fixture,'x')-1.15
        self.phase='travel';self.age=0
    def get_action(self,state):
        if self.phase=='done':return np.zeros(self.action_space.shape,dtype=np.float32)
        self.steps+=1;self.age+=1
        q=np.array([state.get(self.robot,'pos_arm_joint'+str(i)) for i in range(1,8)])
        b=np.array([state.get(self.robot,f) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
        grip=0.
        if self.phase=='travel':
            self.target=self.down
            if np.max(np.abs(self.base-b))<.012 and np.max(np.abs(q-self.target))<.022:
                self.phase='close';self.age=0
        elif self.phase=='close':
            self.target=self.down;grip=1.
            if self.age>=6:self.phase='lift';self.age=0
        elif self.phase=='lift':
            self.target=self.up;grip=1.
            if state.get(self.obj,'z')>.12:self.base=self.staging.copy()
            if (self.age>30 and np.max(np.abs(q-self.target))<.015) or self.age>105:
                if state.get(self.obj,'z')<.1:
                    self.choose(state)
                else:
                    self.phase='place';self.age=0
                    self.base=self.dropbase.copy()
        elif self.phase=='place':
            self.target=self.up;grip=1.
            if np.max(np.abs(self.base-b))<.015 or self.age>100:
                self.phase='release';self.age=0
        elif self.phase=='release':
            self.target=self.up
            if self.age>=5:
                self.phase='retreat';self.age=0;self.base[0]-=.35
        elif self.phase=='retreat':
            self.target=self.up
            if self.age>3 and np.max(np.abs(self.base-b))<.015:
                self.index+=1;self.choose(state)
        error=self.base-b
        error[2]=(error[2]+np.pi)%(2*np.pi)-np.pi
        a=np.r_[np.clip(error,-.1,.1),np.clip((self.target-q)*2,-.1,.1),grip]
        return a.astype(np.float32)
