import numpy as np

class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives): self.action_space=action_space;self.reset(None,None)
 def reset(self,state,info): self.k=0
 def get_action(self,s):
  self.k+=1;a=np.zeros(11,np.float32);r=s.get_object_from_name('robot');b=s.get_object_from_name('box0')
  q=np.array([s.get(r,'joint_%d'%i) for i in range(1,8)],float)
  # Broad-range elbow sweep: the reset angles are lower joint limits.
  goal=np.array([0,.65,-np.pi,1.5,0,-.87,np.pi/2])
  a[3:10]=np.clip(goal-q,-.2,.2)
  a[0]=np.clip(s.get(b,'pose_x')-s.get(r,'pos_base_x'),-.2,.2)
  a[1]=np.clip(s.get(b,'pose_y')-s.get(r,'pos_base_y'),-.2,.2)
  a[10]=1 if self.k<8 else -1
  return a
