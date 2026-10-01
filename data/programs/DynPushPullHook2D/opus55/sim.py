import numpy as np
from env_client import make_env
class S:
    def __init__(self, seed=0, env=None):
        self.env = env or make_env()
        self.obs,_ = self.env.reset(seed=seed); self.t=0; self.done=False
    def g(self,o,f): return float(self.obs.get(self.obs.get_object_from_name(o),f))
    def step(self,a):
        a=np.clip(np.array(a,float),self.env.action_space.low*0.99,self.env.action_space.high*0.99)
        self.obs,r,t,tr,_=self.env.step(a); self.t+=1; self.done=self.done or t; return t
    def rob(self): return [round(self.g('robot',f),3) for f in ['x','y','theta','arm_joint','finger_gap']]
    def pose(self,n): return [round(self.g(n,f),3) for f in ['x','y','theta']]
    def goto(self,x,y,th=None,n=200):
        for i in range(n):
            rx,ry,rt=self.g('robot','x'),self.g('robot','y'),self.g('robot','theta')
            dth=0 if th is None else (th-rt+np.pi)%(2*np.pi)-np.pi
            if abs(x-rx)<1e-3 and abs(y-ry)<1e-3 and abs(dth)<1e-3: return True
            self.step([x-rx,y-ry,dth,0,0])
        return False
