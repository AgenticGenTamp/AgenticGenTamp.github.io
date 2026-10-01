import numpy as np
LIM = np.array([0.03,0.03,0.098174,0.08,0.015])
class Ctl:
    def __init__(self, env, seed):
        self.env=env
        self.obs,self.info=env.reset(seed=seed)
        self.R=self.obs.get_object_from_name('robot')
        self.H=self.obs.get_object_from_name('hook')
        self.t=0; self.term=False
    def g(self,k,o=None): return float(self.obs.get(o if o is not None else self.R,k))
    def pose(self): return np.array([self.g('x'),self.g('y'),self.g('theta'),self.g('arm_joint'),self.g('finger_gap')])
    def step(self,a):
        self.obs,r,te,tr,i=self.env.step(np.array(a,dtype=np.float32)); self.t+=1
        self.term=self.term or te
        return te
    def goto(self,x=None,y=None,th=None,arm=None,gap=None,maxit=400,tol=1e-3):
        for _ in range(maxit):
            p=self.pose()
            tgt=np.array([x if x is not None else p[0], y if y is not None else p[1],
                          th if th is not None else p[2], arm if arm is not None else p[3],
                          gap if gap is not None else p[4]])
            d=tgt-p
            d[2]=(d[2]+np.pi)%(2*np.pi)-np.pi
            if np.all(np.abs(d)<tol): return True
            a=np.clip(d,-LIM,LIM)
            self.step(a)
            if self.term: return True
        return False
    def hookpose(self): return (self.g('x',self.H),self.g('y',self.H),self.g('theta',self.H),self.g('held',self.H))
    def smalls(self):
        return [self.obs.get_object_from_name(n) for n in self.obs.get_object_names() if n.startswith('small')]
    def nright(self): return sum(1 for o in self.smalls() if self.g('x',o)>1.75)
