import math
import numpy as np

class SidePick:
    """Extract a block beside a wall from its accessible side."""
    def __init__(self, space):
        self.space=space
    def reset(self,state):
        r=next(iter(state.get_objects(self.space.get_type('kin_robot'))))
        o=next(iter(state.get_objects(self.space.get_type('target_block'))))
        self.initial=state.get(r,'theta')
        self.sign=1 if state.get(o,'x')<1.6 else -1
        self.angle=math.pi+.15 if self.sign==1 else -.15
        self.phase='rise';self.count=0;self.done=False
    def get_action(self,state):
        r=next(iter(state.get_objects(self.space.get_type('kin_robot'))))
        o=next(iter(state.get_objects(self.space.get_type('target_block'))))
        s=next(iter(state.get_objects(self.space.get_type('target_surface'))))
        g=lambda o,f:state.get(o,f)
        rx,ry=g(r,'x'),g(r,'y');ox,oy=g(o,'x'),g(o,'y')
        x,y,theta,arm,gap=rx,1.3,self.angle,.48,.32
        if self.phase=='rise':
            theta=self.initial;arm=.24
            if ry>1.27:self.phase='above'
        elif self.phase=='above':
            arm=.24 if abs(g(r,'theta')-theta)>.01 else .48
            x=ox+self.sign*(g(o,'width')/2+.6)
            if abs(rx-x)<.01 and abs(g(r,'theta')-theta)<.01 and abs(g(r,'arm_joint')-.48)<.005:
                self.phase='descend';self.pickx=ox+self.sign*(g(o,'width')/2+.53)
        elif self.phase=='descend':
            x=self.pickx;y=.4
            if abs(rx-x)<.01 and abs(ry-y)<.01:self.phase='close';self.count=0
        elif self.phase=='close':
            x=rx-self.sign*.005;y=ry;gap=g(r,'finger_gap')-.001
            self.count+=1
            if g(o,'held'):
                self.phase='drag';self.angle=g(r,'theta')
        elif self.phase=='drag':
            goal=float(np.clip(g(s,'x'),.32-(rx-ox),2.98-(rx-ox)))
            x=rx+np.clip(goal-ox,-.03,.03)
            y=ry;theta=g(r,'theta')-g(o,'theta');gap=g(r,'finger_gap')-.001
            if abs(goal-ox)<.005:self.phase='release';self.count=0
        elif self.phase=='release':
            x=rx;y=ry;theta=g(r,'theta');gap=.32;self.count+=1
            if self.count>15:self.done=True
        return [x-rx,y-ry,theta-g(r,'theta'),arm-g(r,'arm_joint'),gap-g(r,'finger_gap')]
