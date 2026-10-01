import numpy as np
class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives):
  self.a=action_space;r=np.random.default_rng(9182);self.p=np.empty((30,7));self.p[:,0]=r.uniform(-2.2,2.2,30);self.p[:,1]=r.uniform(-2.3,2.3,30);self.p[:,2]=r.uniform(-3.14,3.14,30);self.p[:,3]=r.uniform(-2.58,-.32,30);self.p[:,4]=r.uniform(-2.2,2.2,30);self.p[:,5]=r.uniform(-2.08,1.34,30);self.p[:,6]=r.uniform(-3.14,3.14,30)
 def reset(self,s,info):self.k=0
 def get_action(self,s):
  i=min(self.k//33,29);self.k+=1;a=np.zeros(self.a.shape,dtype=self.a.dtype);a[-1]=1;r=s.get_object_from_name('robot');q=np.array([float(s.get(r,f'pos_arm_joint{j}')) for j in range(1,8)]);e=self.p[i]-q;e[[0,2,4,6]]=(e[[0,2,4,6]]+np.pi)%(2*np.pi)-np.pi;a[3:10]=np.clip(.7*e,-.1,.1);return a
