import numpy as np
class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives):self.a=action_space;self.p=[[2.867,1.088,-.581,1.371,-1.412,2.09,1.108],[-2.835,-1.999,-3.086,1.829,-.526,-2.09,2.195]]
 def reset(self,s,info):self.k=0
 def get_action(self,s):
  self.k+=1;a=np.zeros(self.a.shape,dtype=self.a.dtype);a[-1]=1;r=s.get_object_from_name('robot');q=np.array([float(s.get(r,f'pos_arm_joint{i}')) for i in range(1,8)]);t=np.array(self.p[min(self.k//180,1)]);e=t-q;e[[0,2,4,6]]=(e[[0,2,4,6]]+np.pi)%(2*np.pi)-np.pi;a[3:10]=np.clip(.5*e,-.1,.1);return a
