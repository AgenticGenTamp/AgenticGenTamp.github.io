import numpy as np
class GeneratedApproach:
 def __init__(self,action_space,observation_space,primitives): self.os=observation_space
 def reset(self,s,i):self.t=0;self.r=s.get_objects(self.os.get_type('mujoco_tidybot_robot'))[0]
 def get_action(self,s):
  self.t+=1;a=np.zeros(11);a[10]=1
  return a
