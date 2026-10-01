import numpy as np
class GeneratedApproach:
    def __init__(self, *a, **k): self.t=0
    def reset(self, *a, **k): self.t=0
    def get_action(self, obs=None, *a, **k):
        act=np.zeros(11,dtype=np.float32)
        act[10]= 0.0 if self.t<6 else 1.0
        self.t+=1
        return act
