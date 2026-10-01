import numpy as np
from kin import ik, fk_arm
Rdown=np.array([[0,1,0],[1,0,0],[0,0,-1.]])
J=['joint_%d'%i for i in range(1,8)]
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives): pass
    def reset(self, state, info):
        self.t=0; self.phase=0
    def get_action(self, s):
        r=s.get_object_from_name('robot'); b=s.get_object_from_name('box0')
        bx,by=s.get(b,'pose_x'),s.get(b,'pose_y')
        q=np.array([s.get(r,j) for j in J]); a=np.zeros(11,dtype=np.float32)
        tx,ty=bx-0.6,by
        d=np.array([tx-s.get(r,'pos_base_x'),ty-s.get(r,'pos_base_y')])
        if np.linalg.norm(d)>1e-4:
            a[:2]=np.clip(d,-0.2,0.2); return a
        self.t+=1
        z=max(0.1-0.01*self.t,-0.13)
        qt,_,_=ik(np.array([0.5,0,z]),Rdown,q)
        a[3:10]=np.clip(qt-q,-0.05,0.05)
        return a
