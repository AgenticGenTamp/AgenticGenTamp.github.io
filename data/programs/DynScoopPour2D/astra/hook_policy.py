import math
import numpy as np

class HookPolicy:
    """Grasp the upright, slide its foot under the pile, lift, and pour."""
    def __init__(self,space):
        self.space=space
    def reset(self,s):
        self.r=s.get_objects(self.space.get_type('kin_robot'))[0]
        self.h=s.get_objects(self.space.get_type('hook'))[0]
        self.phase=0;self.age=0;self.stall=0;self.prev=None
        self.hx=s.get(self.h,'x');self.hy=s.get(self.h,'y')
        self.grasp_angle=-1.2 if self.hx>3.4 else -math.pi/2
        self.grasp_x=3.24 if self.hx>3.4 else min(self.hx,3.28)
    def act(self,s):
        g=s.get;r=self.r;h=self.h
        rx,ry,rt,aj,gap=[g(r,f) for f in ('x','y','theta','arm_joint','finger_gap')]
        hx,hy,ht,held=[g(h,f) for f in ('x','y','theta','held')]
        self.age+=1
        axis=1;delta=0
        if self.phase==0:delta=2.4-ry
        elif self.phase==1:axis=0;delta=2.8-rx
        elif self.phase==2:axis=2;delta=self.grasp_angle-rt
        elif self.phase==3:axis=0;delta=self.grasp_x-rx
        elif self.phase==4:delta=self.hy+.70-ry
        elif self.phase==5:axis=4;delta=.08-gap
        elif self.phase==6:delta=1.65-hy
        elif self.phase==7:axis=0;delta=1.42-rx
        elif self.phase==8:axis=2;delta=-math.pi/2-ht
        elif self.phase==9:delta=-.008-hy
        elif self.phase==10:axis=0;delta=.65-hx
        elif self.phase==11:axis=2;delta=-1.77-ht
        elif self.phase==12:delta=1.72-hy
        elif self.phase==13:axis=0;delta=2.7-hx
        elif self.phase==14:axis=2;delta=-.8-ht
        elif self.phase==15:delta=0
        if axis==2:delta=(delta+math.pi)%(2*math.pi)-math.pi
        cur=(rx,ry,rt,aj,gap,hx,hy,ht)
        if self.prev and max(abs(a-b) for a,b in zip(cur,self.prev))<1e-5:self.stall+=1
        else:self.stall=0
        done=abs(delta)<.008 or self.stall>12 or self.age>130
        if self.phase==5:done=bool(held) or self.age>35
        if self.phase==15:done=self.age>20
        if done:
            self.phase+=1;self.age=0;self.stall=0
            if self.phase>15:self.phase=7
        self.prev=cur
        a=np.zeros(5);a[axis]=delta
        return np.clip(a,[-.0299,-.0299,-.0979,-.0799,-.0149],[.0299,.0299,.0979,.0799,.0149])
