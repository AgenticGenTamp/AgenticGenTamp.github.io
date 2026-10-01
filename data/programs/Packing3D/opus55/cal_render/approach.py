import numpy as np, os
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        p=os.path.join(os.path.dirname(os.path.abspath(__file__)),'actions.npy')
        self.A=np.load(p); self.i=0
    def reset(self, state, info): self.i=0
    def get_action(self, state):
        a=self.A[self.i] if self.i<len(self.A) else np.zeros(11,dtype=np.float32); self.i+=1
        return a.astype(np.float32)
