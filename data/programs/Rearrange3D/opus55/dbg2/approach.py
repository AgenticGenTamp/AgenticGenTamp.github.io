import numpy as np
class GeneratedApproach:
    def __init__(self,action_space=None,observation_space=None,primitives=None): pass
    def reset(self,obs,info): self.t=0
    def get_action(self,obs):
        self.t+=1
        a=np.zeros(11,np.float32); a[10]=1.0 if self.t>3 else 0.0
        return a
