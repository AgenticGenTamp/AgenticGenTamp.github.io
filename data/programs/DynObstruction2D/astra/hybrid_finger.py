import math
import numpy as np
from narrow_pick import NarrowPick

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space=observation_space
        self.low=action_space.low*.999
        self.high=action_space.high*.999
    def reset(self,state,info):
        self.phase='rise'
        self.count=0
        self.initial_theta=None
        target=next(iter(state.get_objects(self.space.get_type('target_block'))))
        self.narrow=None
        if state.get(target,'width')<.255:
            self.narrow=NarrowPick(self.space);self.narrow.reset(state)
    def get_action(self,state):
        if self.narrow is not None:
            self.phase=self.narrow.phase
            return np.clip(self.narrow.get_action(state),self.low,self.high).astype(np.float32)
        r=next(iter(state.get_objects(self.space.get_type('kin_robot'))))
        o=next(iter(state.get_objects(self.space.get_type('target_block'))))
        s=next(iter(state.get_objects(self.space.get_type('target_surface'))))
        g=lambda o,f:state.get(o,f)
        rx,ry=g(r,'x'),g(r,'y');ox,oy=g(o,'x'),g(o,'y');sx=g(s,'x')
        if self.count==0:self.direction=1 if sx>ox else -1
        self.count+=1
        if self.initial_theta is None:self.initial_theta=g(r,'theta')
        th=self.initial_theta if self.phase=='rise' else -math.pi/2
        arm=.24 if self.phase=='rise' else .48
        d=self.direction
        behind=np.clip(ox-d*(g(o,'width')/2+.22+.04),.35,2.99)
        x,y=rx,1.3
        if self.phase=='rise':
            if ry>1.27:self.phase='orient'
        elif self.phase=='orient':
            if abs(g(r,'theta')-th)<.015 and abs(g(r,'arm_joint')-.48)<.005:self.phase='across'
        elif self.phase=='across':
            x=behind
            if abs(rx-x)<.015:self.phase='down'
        elif self.phase=='down':
            x=behind;y=.84
            if abs(ry-y)<.015:self.phase='push'
        elif self.phase=='push':
            y=.84
            err=sx-ox
            x=rx+d*min(.01,max(.002,abs(err)*.3)) if d*err>0 else rx
            if abs(err)<.012 or d*err<0:
                self.phase='settle';self.settle_x=rx-d*.09;self.settle_count=0
        elif self.phase=='settle':
            x=self.settle_x;y=.84;self.settle_count+=1
            if self.settle_count>15:
                self.direction=1 if sx>ox else -1
                self.phase='rise'
                self.initial_theta=g(r,'theta')
        a=[x-rx,y-ry,th-g(r,'theta'),arm-g(r,'arm_joint'),.32-g(r,'finger_gap')]
        return np.clip(a,self.low,self.high).astype(np.float32)
