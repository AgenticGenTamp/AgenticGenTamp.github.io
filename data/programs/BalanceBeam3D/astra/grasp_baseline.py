import numpy as np
from scipy.optimize import least_squares
from kinematics_candidate import forward
HOME=np.array([0,-.3491,np.pi,-2.5482,0,-.8727,np.pi/2])
SEED=np.array([0,1.5,np.pi,-1.2,0,-.4,np.pi/2])
BOUNDS=([-6.28,-2.24,-6.28,-2.58,-6.28,-2.09,-6.28],[6.28,2.24,6.28,2.58,6.28,2.09,6.28])
def ik(x,z,q=SEED,mount=.4,extension=.12):
    def fun(q):
        t=forward(q,extension,(0,0,mount))
        return np.r_[5*(t[:3,3]-[x,0,z]),t[:3,2]-[0,0,-1], .0001*(q-SEED)]
    return least_squares(fun,q,bounds=BOUNDS,max_nfev=100).x
class GeneratedApproach:
    mount=.4
    extension=.12
    close=1.
    def __init__(self, action_space, observation_space, primitives):
        self.action_space=action_space
    def reset(self,state,info):
        self.t=0;self.phase=0;self.age=0
        self.qs=[ik(.55,z,mount=self.mount,extension=self.extension) for z in [.15,.018,.22]]
        self.b=np.array([state[0]-.55,state[1],0])
    def get_action(self,state):
        self.t+=1;self.age+=1
        q=self.qs[[0,1,1,2][min(self.phase,3)]]
        if np.max(np.abs(q-state[19:26]))<.015 and np.max(np.abs(self.b-state[16:19]))<.01 or self.age>140:
            if self.phase!=2 or self.age>20:
                self.phase=min(self.phase+1,3);self.age=0
        a=np.zeros(11)
        a[:3]=np.clip(self.b-state[16:19],-.06,.06)
        a[3:10]=np.clip(q-state[19:26],-.1,.1)
        a[10]=self.close if self.phase>=2 else 1-self.close
        return a.astype(np.float32)
