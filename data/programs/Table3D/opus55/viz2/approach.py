import numpy as np
from kin import *
J=[f'joint_{i}' for i in range(1,8)]
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives): pass
    def reset(self, state, info):
        self.t=0
    def get_action(self, state):
        r=state.get_object_from_name('robot'); q=np.array([state.get(r,n) for n in J])
        self.t+=1
        # go to a pose off to the side, gripper pointing sideways for visibility, low
        qt,_=ik(q,np.array([0.5,-0.3,0.25]),down_R(0.0))
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qt-q,-0.4,0.4)
        if self.t>8: a[10]=-1
        return a
