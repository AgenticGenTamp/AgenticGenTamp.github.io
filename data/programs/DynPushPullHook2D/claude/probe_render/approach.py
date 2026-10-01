import numpy as np
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
class GeneratedApproach:
    def __init__(self, action_space, observation_space, primitives): pass
    def reset(self, state, info):
        self.k=0
    def g(self,state,n,f): return float(state.get(state.get_object_from_name(n),f))
    def get_action(self, state):
        self.k+=1
        hx,hy,hth=self.g(state,'hook','x'),self.g(state,'hook','y'),self.g(state,'hook','theta')
        x,y,th=self.g(state,'robot','x'),self.g(state,'robot','y'),self.g(state,'robot','theta')
        u=np.array([np.cos(hth),np.sin(hth)]); nv=np.array([-np.sin(hth),np.cos(hth)])
        a=np.zeros(5)
        if self.k<60:
            a[2]=np.clip(wrap(np.pi+hth-th),-0.0599,0.0599); a[3]=-0.0999; a[4]=-0.0199
            return a.astype(np.float32)
        if not hasattr(self,'stand'):
            self.stand=np.array([hx,hy])+(-0.107-0.275)*u-0.45*nv-0.3*u
        if self.k<160:
            a[0]=np.clip(self.stand[0]-x,-0.0499,0.0499); a[1]=np.clip(self.stand[1]-y,-0.0499,0.0499)
            return a.astype(np.float32)
        if self.k<220:
            a[0]=0.02
            return a.astype(np.float32)
        return a.astype(np.float32)
