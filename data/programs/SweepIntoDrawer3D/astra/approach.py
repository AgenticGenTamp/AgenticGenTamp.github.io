import numpy as np
from kinova import planar_ik

class GeneratedApproach:
    def __init__(self,action_space,observation_space,primitives):
        self.low=np.asarray(action_space.low);self.high=np.asarray(action_space.high)
    def reset(self,state,info):
        self.qdrawer=planar_ik(.6,-.05,-np.pi/2);self.qdrawer[6]+=np.pi/2
        self.qhi=planar_ik(.7,.20)
        self.qlo=planar_ik(.7,.06)
        self.qplace=planar_ik(.7,.10)
        self.stage=0;self.tick=0;self.cube=0;self.done=set();self.tries=np.zeros(5);self.t=0
        self.target=np.array(state[:80]).reshape(5,16)[:,:3].copy()
        self.plan=[(130,2.0,0.,self.qdrawer,0),(45,1.6,0.,self.qdrawer,0),
                   (15,1.6,0.,self.qdrawer,1),(25,2.0,0.,self.qdrawer,1),
                   (12,2.2,0.,self.qdrawer,0),(135,2.2,0.,self.qhi,0)]
    def get_action(self,state):
        s=np.asarray(state);self.t+=1;self.tick+=1
        if self.stage<len(self.plan):
            n,x,y,q,g=self.plan[self.stage]
            if self.tick>=n:self.stage+=1;self.tick=0
        else:
            k=(self.stage-len(self.plan))%6
            cubes=s[:80].reshape(5,16)
            if k==4 and self.tick==1 and cubes[self.cube,2]<.50:
                self.stage+=2;self.tick=1;k=0
            if k==0 and self.tick==1:
                candidates=[i for i in range(5) if cubes[i,2]>.35 and cubes[i,0]<.9]
                if not candidates:self.done=set();candidates=list(range(5))
                self.cube=min(candidates,key=lambda i:(self.tries[i],-cubes[i,0]))
                self.pick=cubes[self.cube,:3].copy();self.tries[self.cube]+=1
            if k==2 and self.tick<=6:self.pick=cubes[self.cube,:3].copy()
            px,py=self.pick[:2]
            if k==0:n,x,y,q,g=20,px+.82,py,self.qhi,0
            elif k==1:n,x,y,q,g=20,px+.82,py,self.qlo,0
            elif k==2:n,x,y,q,g=14,px+.82,py,self.qlo,float(self.tick>6)
            elif k==3:n,x,y,q,g=20,px+.82,py,self.qhi,1
            elif k==4:n,x,y,q,g=22,.89+.4*s[107]+.82,(self.cube-2)*.045,self.qhi,1
            else:n,x,y,q,g=10,.89+.4*s[107]+.82,(self.cube-2)*.045,self.qhi,0
            if k in (0,4) and self.tick>=10:
                if max(abs(s[125]-x),abs(s[126]-y))<.002 and np.max(abs(q-s[128:135]))<.015:n=self.tick
            if k in (1,3) and self.tick>=12:
                if np.max(abs(q-s[128:135]))<.008:n=self.tick
            if self.tick>=n:
                if k==5:self.done.add(self.cube)
                self.stage+=1;self.tick=0
        speed=.03 if self.stage==3 else (.05 if self.stage<3 else .04)
        gain=1. if self.stage<6 else 2.
        a=np.zeros(11);a[:3]=np.clip(np.array([x,y,np.pi])-s[125:128],-speed,speed)
        a[3:10]=np.clip(gain*(q-s[128:135]),-.1,.1);a[10]=g
        return np.clip(a,self.low,self.high).astype(np.float32)
