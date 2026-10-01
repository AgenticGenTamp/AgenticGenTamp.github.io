import numpy as np
class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives):self.a=action_space;self.t=np.array([-1.998,.858,-2.282,-.614,1.223,-2.09,2.965])
 def reset(self,s,info):pass
 def get_action(self,s):
  a=np.zeros(self.a.shape,dtype=self.a.dtype);a[-1]=1;r=s.get_object_from_name('robot');q=np.array([float(s.get(r,f'pos_arm_joint{i}')) for i in range(1,8)]);e=self.t-q;e[[0,2,4,6]]=(e[[0,2,4,6]]+np.pi)%(2*np.pi)-np.pi;a[3:10]=np.clip(.5*e,-.1,.1);return a
