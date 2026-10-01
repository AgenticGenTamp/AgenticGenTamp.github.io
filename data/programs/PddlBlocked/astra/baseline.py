import math
import numpy as np

class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        self.space=observation_space
        self.fs=['base_x','base_y','base_rot']+['joint_'+str(i) for i in range(1,8)]
    def reset(self,state,info):
        self.r=state.get_object_from_name('robot')
        self.q=np.array([0,.45,0,-.4,0,-.05,0])
        self.g=self.pos(state,'green0')
        self.b=self.pos(state,'blocker')
        d=self.g[:2]-self.b[:2]; self.yaw=math.atan2(d[1],d[0]);self.d=d/np.linalg.norm(d)
        self.R=np.array([[math.cos(self.yaw),-math.sin(self.yaw)],[math.sin(self.yaw),math.cos(self.yaw)]])
        self.offset=self.R@np.array([.94,.188]);self.queue=[];self.stage=0;self.stuck=0;self.prev=None;self.ticks=0
        # Travel around table before approaching the pen along its opening.
        pre=self.b[:2]-self.offset-.2*self.d
        c=np.array([4.5,0.]); angle=math.atan2(pre[1],pre[0]-4.5)
        diff=(angle-math.pi+math.pi)%(2*math.pi)-math.pi
        for a in np.linspace(math.pi,math.pi+diff,max(2,int(abs(diff)/.25)+1)):
            self.add(c+1.7*np.array([math.cos(a),math.sin(a)]),1)
        self.add(pre,1);self.add(self.b[:2]-self.offset,-1)
    def pos(self,s,name):
        o=s.get_object_from_name(name);return np.array([s.get(o,'pose_'+f) for f in 'xyz'])
    def add(self,xy,grip=0):
        self.queue.append((np.r_[xy,self.yaw,self.q],grip))
    def get_action(self,s):
        self.ticks+=1
        p=np.array([s.get(self.r,f) for f in self.fs]);held=s.get(self.r,'grasp_active')>.5
        if not self.queue:
            if self.stage==0:
                self.stage=1;self.add(p[:2]-.24*self.d,1 if not held else 0)
                self.add(p[:2]-.24*self.d,1)
            elif self.stage==1:
                self.stage=2;self.add(self.g[:2]-self.offset-.25*self.d,1);self.add(self.g[:2]-self.offset,-1)
            elif self.stage==2:
                self.stage=3;self.add(p[:2]-.3*self.d,0)
                raised=np.r_[p[:2]-.3*self.d,self.yaw,self.q.copy()];raised[4]-=.3;self.queue.append((raised,0))
            elif self.stage==3:
                self.stage=4
                v=p.copy();v[0]=3.1;self.queue.append((v.copy(),0));v[1]=-.5;self.queue.append((v.copy(),0));v[2]=0;self.queue.append((v.copy(),0))
            elif self.stage==4:
                self.stage=5;plate=self.pos(s,'plate');block=self.pos(s,'green0');v=p.copy();v[:2]+=plate[:2]-block[:2];self.queue.append((v,1))
            else:
                return np.zeros(11,dtype=np.float32)
        target,grip=self.queue[0];d=target-p;d[[2,7,9]]=(d[[2,7,9]]+math.pi)%(2*math.pi)-math.pi
        a=np.zeros(11);a[:10]=np.clip(d,-.2,.2)
        # Apply close/open only when arriving, keep fingers open en route to grasp.
        if np.max(abs(d))<.20001:a[10]=grip
        elif grip:a[10]=1 if not held else 0
        if np.max(abs(d))<1e-4:
            self.queue.pop(0)
        return a.astype(np.float32)
