import math
import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space=observation_space
        self.low=action_space.low*.999
        self.high=action_space.high*.999
    def reset(self,state,info):
        self.phase='rise'
        self.count=0
        self.initial_theta=None
    def get_action(self,state):
        r=next(iter(state.get_objects(self.space.get_type('kin_robot'))))
        o=next(iter(state.get_objects(self.space.get_type('target_block'))))
        s=next(iter(state.get_objects(self.space.get_type('target_surface'))))
        g=lambda o,f:state.get(o,f)
        rx,ry=g(r,'x'),g(r,'y');ox,oy=g(o,'x'),g(o,'y');sx=g(s,'x')
        if self.count==0:self.direction=1 if sx>ox else -1
        self.count+=1
        if self.initial_theta is None:self.initial_theta=g(r,'theta')
        th=self.initial_theta if self.phase=='rise' else math.pi/2
        arm=.24
        d=self.direction
        behind=np.clip(ox-d*(g(o,'width')/2+g(r,'base_radius')+.04),.35,2.99)
        x,y=rx,1.3
        if self.phase=='rise':
            if ry>1.27:self.phase='across'
        elif self.phase=='across':
            x=behind
            if abs(rx-x)<.015:self.phase='down'
        elif self.phase=='down':
            x=behind;y=.375
            if abs(ry-y)<.015:self.phase='push'
        elif self.phase=='push':
            y=.375
            err=sx-ox
            x=rx+d*min(.01,max(.002,abs(err)*.3)) if d*err>0 else rx
            if d*err<-.025:
                self.direction=-d;self.phase='rise'
        a=[x-rx,y-ry,th-g(r,'theta'),arm-g(r,'arm_joint'),.32-g(r,'finger_gap')]
        return np.clip(a,self.low,self.high).astype(np.float32)
