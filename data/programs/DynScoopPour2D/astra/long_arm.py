import math
import numpy as np
from hook_policy import HookPolicy

class GeneratedApproach:
    """Feedback scooping with a physical hook for sparse floor objects."""
    def __init__(self, action_space, observation_space, primitives):
        self.space=observation_space
        self.low=np.maximum(action_space.low,[-.03,-.03,-.098,-.08,-.015])
        self.high=np.minimum(action_space.high,[.03,.03,.098,.08,.015])
    def reset(self,state,info):
        self.robot=state.get_objects(self.space.get_type('kin_robot'))[0]
        self.hook=state.get_objects(self.space.get_type('hook'))[0]
        self.smalls=[o for typ in ("small_circle","small_square") for o in state.get_objects(self.space.get_type(typ))]
        self.hook_controller=HookPolicy(self.space) if len(self.smalls)<10 else None
        if self.hook_controller:self.hook_controller.reset(state)
        self.phase=0
        self.steps=0
        self.age=0
        self.stall=0
        self.last=None
        self.cycle=0
        self.travel_y=2.6
        self.arm_ready=False
    def get_action(self,state):
        if self.hook_controller:
            action=self.hook_controller.act(state)
            self.phase=self.hook_controller.phase
            return action
        if self.phase==2 and not self.arm_ready:
            d=.4-state.get(self.robot,"arm_joint")
            if abs(d)<.003:self.arm_ready=True
            a=np.zeros(5);a[3]=np.clip(d,-.0799,.0799);return a
        self.steps+=1;self.age+=1
        g=state.get;r=self.robot
        cur=np.array([g(r,f) for f in ('x','y','theta','arm_joint','finger_gap')])
        x,y,t,a,gap=cur
        if self.phase==0:target=[x,self.travel_y,t,.2,.25]
        elif self.phase==1:target=[x,y,0,.2,.25]
        elif self.phase==2:target=[.23,self.travel_y,0,.2,.25]
        elif self.phase==3:target=[.23,.2,0,.2,.25]
        elif self.phase==4:target=[1.16,.2,0,.2,.25]
        elif self.phase==5:target=[x,1.85,0,.2,.25]
        elif self.phase==6:target=[2.7,1.85,0,.2,.25]
        elif self.phase==7:target=[x,y,1.57,.2,.25]
        else:target=cur.copy()
        delta=np.asarray(target)-cur
        delta[2]=(delta[2]+math.pi)%(2*math.pi)-math.pi
        axis=[1,2,0,1,0,1,0,2,0][self.phase]
        # Tiny corrections on a second axis can reject an entire floor move.
        amount=delta[axis]
        delta[:]=0
        delta[axis]=amount
        if self.last is not None and np.max(np.abs(cur-self.last))<1e-5:self.stall+=1
        else:self.stall=0
        done=np.max(np.abs(delta))<.006 or self.stall>7 or self.age>180
        if self.phase==8:done=self.age>=20
        if done:
            self.phase+=1
            self.age=0;self.stall=0
            if self.phase>8:self.phase=1;self.cycle+=1;self.travel_y=1.85
        self.last=cur.copy()
        action=np.clip(delta,self.low,self.high)
        return action
