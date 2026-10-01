from env_client import make_env
import numpy as np, math
class P:
    def __init__(self, seed=0):
        self.env = make_env(); self.obs,_ = self.env.reset(seed=seed)
        self.R = self.obs.get_object_from_name('robot')
    def g(self,k,o=None): return float(self.obs.get(o if o is not None else self.R,k))
    def og(self,name,k): return float(self.obs.get(self.obs.get_object_from_name(name),k))
    def step(self,a,n=1):
        for _ in range(n): self.obs,_,_,_,_ = self.env.step(np.array(a,dtype=np.float64))
    def pose(self): return (self.g('x'),self.g('y'),self.g('theta'))
    def drive(self,ax,ay,maxn=400):
        prev=self.pose()[:2]
        for i in range(maxn):
            self.step([ax,ay,0,0,0]); cur=self.pose()[:2]
            if abs(cur[0]-prev[0])<1e-6 and abs(cur[1]-prev[1])<1e-6: return i
            prev=cur
        return maxn
    def moveto(self,tx,ty,tol=0.002,maxn=600):
        for i in range(maxn):
            x,y,_=self.pose()
            dx,dy=tx-x,ty-y
            if abs(dx)<tol and abs(dy)<tol: return True
            self.step([float(np.clip(dx,-0.03,0.03)),float(np.clip(dy,-0.03,0.03)),0,0,0])
        return False
    def setth(self,t,maxn=300):
        for _ in range(maxn):
            d=math.atan2(math.sin(t-self.g('theta')),math.cos(t-self.g('theta')))
            if abs(d)<2e-3: return True
            b=self.g('theta'); self.step([0,0,float(np.clip(d,-0.098,0.098)),0,0])
            if abs(self.g('theta')-b)<1e-9: return False
        return True
    def grip(self,d,n=40): self.step([0,0,0,0,d],n)
    def arm(self,d,n=40): self.step([0,0,0,d,0],n)
    def close(self): self.env.close()
