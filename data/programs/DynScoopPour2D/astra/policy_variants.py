import math
import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space=observation_space
        self.low=np.maximum(action_space.low,[-.03,-.03,-.098,-.08,-.015])
        self.high=np.minimum(action_space.high,[.03,.03,.098,.08,.015])
    def reset(self,state,info):
        self.robot=state.get_objects(self.space.get_type('kin_robot'))[0]
        self.hook=state.get_objects(self.space.get_type('hook'))[0]
        self.smalls=[o for typ in ("small_circle","small_square") for o in state.get_objects(self.space.get_type(typ))]
        self.scoop_theta=-.16 if len(self.smalls)<10 else 0.0
        self.phase=0
        self.steps=0
        self.age=0
        self.stall=0
        self.last=None
        self.cycle=0
        self.travel_y=2.6
        self.tipped=False
    def get_action(self,state):
        self.steps+=1;self.age+=1
        g=state.get;r=self.robot
        cur=np.array([g(r,f) for f in ('x','y','theta','arm_joint','finger_gap')])
        x,y,t,a,gap=cur
        if self.phase==0:target=[x,self.travel_y,t,.2,.25]
        elif self.phase==1:target=[x,y,self.scoop_theta,.2,.25]
        elif self.phase==2:target=[.23,self.travel_y,0,.2,.25]
        elif self.phase==3:target=[.23,.2,0,.2,.25]
        elif self.phase==4:target=[1.36,.2,0,.2,.25]
        elif self.phase==5:target=[x,1.8,0,.2,.25]
        elif self.phase==6:target=[2.45,1.8,0,.2,.25]
        elif self.phase==7:target=[x,y,1.57,.2,.25]
        else:target=cur.copy()
        delta=np.asarray(target)-cur
        delta[2]=(delta[2]+math.pi)%(2*math.pi)-math.pi
        axis=[1,2,0,1,0,1,0,2,0][self.phase]
        scoop_tip=self.phase==5 and len(self.smalls)<10 and not self.tipped
        if scoop_tip:
            axis=2
            delta[2]=.55-t
        amount=delta[axis]
        delta[:]=0
        delta[axis]=amount
        if self.last is not None and np.max(np.abs(cur-self.last))<1e-5:self.stall+=1
        else:self.stall=0
        done=np.max(np.abs(delta))<.006 or self.stall>7 or self.age>180
        if self.phase==8:done=self.age>=20
        if done:
            if scoop_tip:self.tipped=True
            else:self.phase+=1
            self.age=0;self.stall=0
            if self.phase>8:self.phase=1;self.cycle+=1;self.travel_y=1.8;self.tipped=False
        self.last=cur.copy()
        action=np.clip(delta,self.low,self.high)
        return action
