import math
import numpy as np
from kinematics import fk, solve

class GeneratedApproach:
    def __init__(self,action_space,observation_space,primitives):
        self.space=observation_space
        self.fs=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
    def pos(self,s,name):
        o=s.get_object_from_name(name);return np.array([s.get(o,'pose_'+f) for f in 'xyz'])
    def reset(self,s,info):
        self.r=s.get_object_from_name('robot');self.stage=0;self.queue=[];self.ticks=0;self.prev=None;self.stuck=0
        self.g=self.pos(s,'green0');self.b=self.pos(s,'blocker');self.d=(self.g-self.b)/np.linalg.norm(self.g-self.b);self.yaw=math.atan2(self.d[1],self.d[0])
        self.candidates=[]
        for xy in [[3.73,self.g[1]+v] for v in [0,-.3,.3,.6]]+[[self.g[0]+v,1.02] for v in [0,-.35,.35]]:
            if xy[0]>4.98:continue
            cfg,err=solve(self.g-.025*self.d,self.yaw,xy)
            if err<.015:
                cr,er=solve(self.b-.025*self.d,self.yaw,xy,initial=cfg)
                cp,ep=solve(self.b-.17*self.d,self.yaw,xy,initial=cr)
                if max(er,ep)<.015:
                    self.candidates.append((cfg,cr,cp))
        self.attempt=0;self.setup(s)
    def add(self,v,grip=0):self.queue.append((np.array(v).copy(),grip))
    def setup(self,s):
        p=np.array([s.get(self.r,f) for f in self.fs]);self.queue=[];self.stage=0
        if not self.candidates:
            self.add(p);return
        self.cg,self.cr,self.cp=self.candidates[self.attempt%len(self.candidates)]
        self.base=self.cp[:2].copy();self.north=self.base[1]>.95
        self.high,err=solve(self.b-.17*self.d+np.array([0,0,.3]),self.yaw,self.base,initial=self.cp)
        v=p.copy();v[0]=3.1;self.add(v,1)
        if self.north:v[1]=1.4;self.add(v,1)
        else:v[1]=self.base[1];self.add(v,1)
        v[2:]=self.high[2:];self.add(v,1)
        if self.north:v[0]=self.base[0];self.add(v,1)
        v[:2]=self.base;self.add(v,1)
        self.add(self.high,1);self.add(self.cp,1);self.add(self.cr,-1)
    def nextstage(self,s,p,held):
        if self.stage==0:
            if not held:
                self.attempt+=1;self.setup(s);return
            self.stage=1;self.add(self.cp,0);self.add(self.high,0)
            v=self.high.copy();v[1 if self.north else 0]+=.55 if self.north else -.55;self.add(v,1)
        elif self.stage==1:
            self.stage=2;self.add(self.high,1);self.add(self.cp,1);self.add(self.cr,1);self.add(self.cg,-1)
        elif self.stage==2:
            if not held:
                self.stage=1;self.add(self.cp,1);return
            self.stage=3;self.add(self.cr,0);self.add(self.cp,0);self.add(self.high,0)
        elif self.stage==3:
            self.stage=4
            v=p.copy()
            if self.north:v[1]=1.4;self.add(v,0)
            v[0]=3.1;self.add(v,0)
            v[2:]=np.r_[0,[0,.15,0,-.4,0,-.05,0]];self.add(v,0)
            v[1]=-.5;self.add(v,0)
        elif self.stage==4:
            self.stage=5;v=p.copy();v[:2]+=self.pos(s,'plate')[:2]-self.pos(s,'green0')[:2];self.add(v,1)
        else:self.add(p,1)
    def get_action(self,s):
        self.ticks+=1;p=np.array([s.get(self.r,f) for f in self.fs]);held=s.get(self.r,'grasp_active')>.5
        if self.prev is not None and np.max(abs(p-self.prev))<1e-5:self.stuck+=1
        else:self.stuck=0
        self.prev=p.copy()
        if not self.queue:self.nextstage(s,p,held)
        target,grip=self.queue[0];d=target-p;d[[2,7,9]]=(d[[2,7,9]]+math.pi)%(2*math.pi)-math.pi
        a=np.zeros(11);a[:10]=d*min(1.,.2/max(1e-9,np.max(abs(d))))
        if np.max(abs(d))<.20001:a[10]=grip
        elif grip:a[10]=1 if not held else 0
        if np.max(abs(d))<1e-4:self.queue.pop(0)
        return a.astype(np.float32)
