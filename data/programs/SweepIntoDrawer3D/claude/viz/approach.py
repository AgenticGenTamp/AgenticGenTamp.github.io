import numpy as np
from ik import ik
RDOWN=np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=float)
class GeneratedApproach:
    def __init__(self, action_space=None, observation_space=None, primitives=None): self.t=0
    def reset(self,state,info): self.t=0; self.qd=None
    def get_action(self,state):
        self.t+=1
        if self.t<40: z=0.35
        else: z=max(0.20, 0.35-0.005*(self.t-40))
        q=state[128:135]
        qd,_=ik(np.array([0.500,0.074,z]),RDOWN,q)
        a=np.zeros(11); a[3:10]=np.clip(qd-q,-0.1,0.1); a[10]=1.0
        return a
