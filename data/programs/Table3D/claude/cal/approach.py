import numpy as np
from fk import fk
from ik2 import solve_pos_axis
J=["joint_%d"%i for i in range(1,8)]
BZ=0.4; GRASP_OFF=0.19
class GeneratedApproach:
    def __init__(self, action_space=None, observation_space=None, primitives=None): pass
    def reset(self,state,info):
        c=state.get_object_from_name("cube0")
        cp=np.array([float(state.get(c,f)) for f in ["pose_x","pose_y","pose_z"]])-np.array([0,0,BZ])
        self.t1=cp+np.array([0,0,GRASP_OFF+0.15])
        self.t2=cp+np.array([0,0,GRASP_OFF])
        self.q1,pe,ae=solve_pos_axis(self.t1,0.0)
        self.q2,pe2,ae2=solve_pos_axis(self.t2,0.0,q0=self.q1)
        print("ik",pe,ae,pe2,ae2)
        self.phase=0; self.k=0
    def get_action(self,state):
        r=state.get_object_from_name("robot")
        q=np.array([float(state.get(r,f)) for f in J])
        self.k+=1
        qt = self.q1 if self.phase==0 else self.q2
        d=qt-q
        if np.max(np.abs(d))<3e-3:
            if self.phase==0: self.phase=1
            else:
                a=np.zeros(11); a[10]=-1.0; return a
        a=np.zeros(11); a[3:10]=np.clip(d,-0.15,0.15)
        return a
