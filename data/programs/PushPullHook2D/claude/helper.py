"""Shared helpers for exploration scripts (NOT imported by approach.py)."""
import numpy as np
from env_client import make_env

IDX = dict(rx=0,ry=1,rth=2,brad=3,arm=4,armlen=5,vac=6,gh=7,gw=8,
           hx=9,hy=10,hth=11,hw=17,hl1=18,hl2=19,
           mx=20,my=21,mr=28,tx=29,ty=30,tr=37)

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
def R(t):
    c,s=np.cos(t),np.sin(t); return np.array([[c,-s],[s,c]])

class Sim:
    """Convenience wrapper: obs cached, goto controller."""
    def __init__(self, seed=42):
        self.env=make_env()
        self.obs,_=self.env.reset(seed=seed)
        self.n=0; self.term=False
    def step(self,a):
        self.obs,r,t,tr,i=self.env.step(np.array(a,dtype=np.float32))
        self.n+=1; self.term=t
        return self.obs
    def goto(self,tx,ty,tth,tarm,vac,maxn=400):
        """Returns steps used; negative if it got stuck."""
        k=0; stuck=0
        while k<maxn and not self.term:
            o=self.obs
            dx=np.clip(tx-o[0],-0.05,0.05); dy=np.clip(ty-o[1],-0.05,0.05)
            dth=np.clip(wrap(tth-o[2]),-0.196,0.196); da=np.clip(tarm-o[4],-0.1,0.1)
            if max(abs(dx),abs(dy),abs(dth),abs(da))<1e-4: break
            p=o[[0,1,2,4]].copy(); self.step([dx,dy,dth,da,vac]); k+=1
            if np.abs(self.obs[[0,1,2,4]]-p).max()<1e-7:
                stuck+=1
                if stuck>3: return -k
            else: stuck=0
        return k
    def grasp_hook(self, d=1.15, standoff=0.30):
        """Grasp long side of hook at distance d from corner. Returns True/False."""
        o=self.obs; th=o[11]; C=o[9:11].copy()
        a=np.array([-np.cos(th),-np.sin(th)]); nh=np.array([-np.sin(th),np.cos(th)])
        P=C+a*d
        s=np.sign(np.dot(o[:2]-P,nh)) or 1.0
        n=nh*s
        rp=P+n*standoff; face=np.arctan2(-n[1],-n[0])
        self.goto(o[0],o[1],face,0.1,0.0)
        self.goto(rp[0],rp[1],face,0.1,0.0)
        self.goto(rp[0],rp[1],face,0.2,0.0)
        for i in range(40):
            pp=self.obs[:2].copy(); self.step([-n[0]*0.01,-n[1]*0.01,0,0,0.0])
            if np.linalg.norm(self.obs[:2]-pp)<1e-7: break
        self.step([0,0,0,0,1.0])
        # verify by tiny move
        return True
    def close(self): self.env.close()
