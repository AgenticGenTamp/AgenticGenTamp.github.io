import numpy as np
class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives):self.a=action_space;self.p=[[0,.329,0,.329,0,0,1.57],[0,-.5,0,1,0,0,1.57],[0,1,0,-.5,0,0,1.57],[0,-1,0,-1,0,0,1.57],[0,1,0,1,0,0,1.57]]
 def reset(self,s,info):self.k=0
 def get_action(self,s):
  self.k+=1;a=np.zeros(self.a.shape,dtype=self.a.dtype);a[-1]=1;r=s.get_object_from_name('robot');q=np.array([float(s.get(r,f'pos_arm_joint{i}')) for i in range(1,8)]);t=np.array(self.p[min(self.k//100,4)]);e=(t-q+np.pi)%(2*np.pi)-np.pi;a[3:10]=np.clip(.4*e,-.1,.1);return a
