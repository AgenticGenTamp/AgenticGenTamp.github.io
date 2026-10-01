import numpy as np
LIM = np.array([0.03,0.03,0.098174,0.08,0.015])
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives):
        pass
    def reset(self, state, info):
        self.t=0
        self.wps=[(2.6,2.0,0.0,0.2,0.25),(2.6,2.0,0.0,0.4,0.25),(2.6,2.0,-1.5708,0.4,0.25),
                  (3.0,0.6,-1.5708,0.4,0.25),(3.0,0.35,-1.5708,0.4,0.25),(3.0,0.35,-1.5708,0.4,0.08)]
        self.i=0; self.hold=0
    def get_action(self, state):
        R=state.get_object_from_name('robot')
        p=np.array([state.get(R,'x'),state.get(R,'y'),state.get(R,'theta'),state.get(R,'arm_joint'),state.get(R,'finger_gap')],dtype=float)
        if self.i>=len(self.wps): return np.zeros(5,dtype=np.float32)
        t=np.array(self.wps[self.i])
        d=t-p; d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
        if np.all(np.abs(d)<2e-3):
            self.i+=1; return np.zeros(5,dtype=np.float32)
        self.hold+=1
        if self.hold>300: self.i+=1; self.hold=0
        return np.clip(d,-LIM,LIM).astype(np.float32)
