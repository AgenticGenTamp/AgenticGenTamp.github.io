"""Helpers for exploration scripts ONLY (never imported by approach.py)."""
import numpy as np
from env_client import make_env

LO=np.array([-0.05,-0.05,-0.19634954084936207,-0.1,-0.02])
HI=-LO
STEP=np.array([0.049,0.049,0.19,0.099,0.019])

class Sim:
    def __init__(self, seed=42, object_count=None):
        self.env=make_env()
        kw={}
        if object_count is not None: kw['options']={'object_count':object_count}
        self.obs,self.info=self.env.reset(seed=seed,**kw)
        self.t=0; self.term=False; self.trunc=False
    def rv(self,f):
        o=self.obs.get_object_from_name('robot'); return float(self.obs.get(o,f))
    def bv(self,n,f):
        o=self.obs.get_object_from_name(n); return float(self.obs.get(o,f))
    def names(self): return sorted(self.obs.get_object_names())
    def obstructions(self): return [n for n in self.names() if n.startswith('obstruction')]
    def act(self,a,n=1):
        for _ in range(n):
            self.obs,r,self.term,self.trunc,i=self.env.step(np.clip(np.array(a,dtype=np.float64),LO,HI))
            self.t+=1
            if self.term or self.trunc: return True
        return self.term
    def goto(self,x=None,y=None,th=None,arm=None,gap=None,steps=400,tol=2e-3):
        """Move toward targets simultaneously (careful: no collision avoidance)."""
        for _ in range(steps):
            a=[0.,0.,0.,0.,0.]; done=True
            for i,(tgt,f) in enumerate([(x,'x'),(y,'y'),(th,'theta'),(arm,'arm_joint'),(gap,'finger_gap')]):
                if tgt is None: continue
                d=tgt-self.rv(f); a[i]=np.clip(d,-STEP[i],STEP[i])
                if abs(d)>tol: done=False
            if done: return True
            if self.act(a): return True
        return False
    def close(self): self.env.close()
