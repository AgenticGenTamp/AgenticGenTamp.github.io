import numpy as np
from kin import *
J=[f'joint_{i}' for i in range(1,8)]
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives): pass
    def reset(self, state, info):
        self.t=0
    def get_action(self, state):
        r=state.get_object_from_name('robot'); q=np.array([state.get(r,n) for n in J])
        c=state.get_object_from_name('cube0')
        x,y=state.get(c,'pose_x'),state.get(c,'pose_y')
        z=max(0.25-0.01*self.t,0.16)
        self.t+=1
        qt,_=ik(q,np.array([x,y,z]),down_R(np.pi/2))
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qt-q,-0.4,0.4)
        if self.t>12: a[10]=-1
        return a
