import numpy as np
class GeneratedApproach:
 def __init__(self, action_space, observation_space, primitives): self.t=0
 def reset(self,state,info): self.t=0
 def get_action(self,s):
  self.t+=1; a=np.zeros(11,np.float32); a[10]=0
  if self.t<18: target=[1.30,-.38]; a[:2]=np.clip((np.array(target)-s[125:127])*.5,-.04,.04);a[4]=.02
  elif self.t<25: pass
  elif self.t<32: a[10]=1
  else: a[0]=.06;a[10]=1
  return a
