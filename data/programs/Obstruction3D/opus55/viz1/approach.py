import numpy as np
from kin import *
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives): pass
    def reset(self, state, info):
        self.t=0
    def get_action(self, s):
        self.t+=1
        R=s.get_object_from_name('robot'); tb=s.get_object_from_name('target_block')
        q=np.array([s.get(R,f'joint_{i}') for i in range(1,8)])
        base=(s.get(R,'pos_base_x'),s.get(R,'pos_base_y'),s.get(R,'pos_base_rot'))
        qt,e=ik(base,q,np.array([s.get(tb,'pose_x'),s.get(tb,'pose_y'),0.25]),down_R(0))
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qt-q,-0.2,0.2)
        if self.t>12: a[:]=0; a[10]=-1
        return a
