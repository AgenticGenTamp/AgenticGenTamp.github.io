import numpy as np
from kinematics import grasp_joints

class GeneratedApproach:
    """Feedback pick-and-place with calibrated downward grasps."""
    def __init__(self, action_space, observation_space, primitives):
        self.low = np.asarray(action_space.low)
        self.high = np.asarray(action_space.high)
        # Joint targets use grasp height relative to the arm mount.
        self.highq=grasp_joints(.27)
        self.lowq=grasp_joints(.12)
    def reset(self, state, info):
        self.highq=grasp_joints(.27)
        self.lowq=grasp_joints(.12)
        self.slide=False
        self.offset=.75
        self.spacing=.105
        self.phase=0
        self.age=0
        self.item=0
        self.retries=0
        self.base=np.array([min(state[93],-.75),state[94],0.])
        self.home=np.asarray(state[96:103]).copy()
        self.goal=np.asarray(state[:3]).copy()
    def select_grasp(self, s, idx):
        w,x,y,z=s[idx+3:idx+7]
        r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                    [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                    [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
        self.highq=grasp_joints(.27)
        self.lowq=grasp_joints(.12)
        self.offset=.75 if idx==16 else .77
        self.spacing=.105
        self.slide=False
        if s[idx+2]<.5:
            self.lowq=grasp_joints(np.clip(s[idx+2]-.392,.075,.12))
            self.offset=.77
        if abs(r[2,2])<.8:
            if idx==16:
                axes=[j for j in range(3) if abs(r[2,j])<.8]
                axis=min(axes,key=lambda j:s[idx+13+j])
                direction=r[:2,axis]
            else:
                direction=np.array([-r[1,2],r[0,2]])
            angle=np.arctan2(direction[1],direction[0])
            wrist=angle%np.pi
            self.highq[-1]=wrist
            self.lowq=grasp_joints(np.clip(s[idx+2]-(.412 if idx==32 else .392),.075,.12))
            self.lowq[-1]=wrist
            self.offset=.77
            self.spacing=min(.13,max(.105,.072+.5*np.dot(abs(r[1]),s[idx+13:idx+16])))

    def next(self):
        self.phase+=1
        self.age=0
    def get_action(self, state):
        s=np.asarray(state)
        self.age+=1
        idx=16 if self.item==0 else 32
        q=self.highq
        grip=0.
        if self.phase==0:
            q=self.home
            if np.max(np.abs(s[93:96]-self.base))<.01 or self.age>25:self.next()
        elif self.phase==1:
            if np.max(np.abs(s[96:103]-q))<.008 or self.age>120:self.next()
        elif self.phase==2:
            # Adapt the finger direction and depth for objects lying on their side.
            if self.age==1:self.select_grasp(s,idx)
            q=self.highq
            self.base=np.array([s[idx]-self.offset,s[idx+1]-.001,0.])
            if (np.max(np.abs(s[93:96]-self.base))<.004 and np.max(np.abs(s[96:103]-q))<.01) or self.age>90:self.next()
        elif self.phase==3:
            q=self.lowq
            if np.max(np.abs(s[96:103]-q))<.008 or self.age>65:self.next()
        elif self.phase==4:
            q=self.lowq;grip=1.
            if self.age>=12:self.next()
        elif self.phase==5:
            grip=1.
            if self.slide:q=self.lowq
            if np.max(np.abs(s[96:103]-q))<.01 or self.age>70:
                if s[idx+2]<.59 and self.retries<2 and not self.slide:
                    self.retries+=1;self.phase=2;self.age=0
                else:
                    self.base=s[93:96].copy()
                    # Correct placement using the observed held-object position.
                    self.base[:2]+=self.goal[:2]+np.array([0,self.spacing if self.item==0 else -self.spacing])-s[idx:idx+2]
                    self.base[2]=0.
                    self.next()
        elif self.phase==6:
            grip=1.
            if self.slide:q=self.lowq
            if np.max(np.abs(s[93:96]-self.base))<.004 or self.age>60:self.next()
        elif self.phase==7:
            q=self.lowq;grip=1.
            desired=self.goal[:2]+np.array([0,self.spacing if self.item==0 else -self.spacing])
            if not self.slide:
                self.base[:2]+=np.clip((desired-s[idx:idx+2])*.25,-.003,.003)
            if np.max(np.abs(s[96:103]-q))<.01 or self.age>65:self.next()
        elif self.phase==8:
            q=self.lowq
            if self.age>=12:self.next()
        elif self.phase==9:
            if np.max(np.abs(s[96:103]-q))<.01 or self.age>65:
                self.item=1-self.item;self.retries=0;self.phase=2;self.age=0
                self.goal=s[:3].copy()
        a=np.clip(np.r_[self.base,q,grip]-s[93:104],self.low,self.high)
        speed=.018 if self.slide and self.phase in (5,6,7) else .045
        a[:2]=np.clip(a[:2],-speed,speed)
        a[-1]=grip
        return a.astype(np.float32)
