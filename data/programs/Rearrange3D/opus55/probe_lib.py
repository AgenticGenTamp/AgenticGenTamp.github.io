import numpy as np
from env_client import make_env
import kin
np.set_printoptions(precision=4,suppress=True,linewidth=200)
def Rdown(psi):
    c,s=np.cos(psi),np.sin(psi)
    return np.array([[c,s,0],[s,-c,0],[0,0,-1.]])
class P:
    def __init__(self,seed=0):
        self.env=make_env(); self.obs,_=self.env.reset(seed=seed)
        self.QT=self.obs[96:103].astype(float).copy(); self.grip=0.0
    def step(self,a):
        a=np.asarray(a,np.float32); a[3:10]=np.clip(a[3:10],-.1,.1)
        self.QT+=0.25*a[3:10]
        self.obs,r,te,tr,_=self.env.step(a); return r,te
    def goto(self,q_t,maxsteps=300,tol=0.002):
        bias=np.zeros(7)
        for i in range(maxsteps):
            e=q_t-self.obs[96:103]
            if np.max(np.abs(q_t-self.QT))<0.03: bias+=0.3*e
            a=np.zeros(11); a[3:10]=4*(q_t+bias-self.QT); a[10]=self.grip
            self.step(a)
            if np.max(np.abs(q_t-self.obs[96:103]))<tol: break
    def ta(self,pw):
        b=self.obs[93:96]
        return kin.rotz(-b[2])@(np.array(pw)-np.array([b[0],b[1],0]))-kin.MOUNT
    def ik(self,pw,psi=0.0,q0=None):
        b=self.obs[93:96]
        q,e=kin.ik_arm(self.ta(pw),kin.rotz(-b[2])@Rdown(psi),self.QT if q0 is None else q0,iters=300)
        return q
    def fk(self,q=None):
        return kin.fk_world(self.obs[93:96],self.obs[96:103] if q is None else q)[0]
    def line(self,p0,p1,psi=0.0,ds=0.002,watch=None,thr=0.002,qthr=0.02):
        """move grasp point along line; stop at contact. returns commanded point at contact"""
        p0=np.array(p0,float);p1=np.array(p1,float)
        n=int(np.linalg.norm(p1-p0)/ds)+1
        q=self.QT.copy(); bias=self.QT-self.obs[96:103]
        ref=self.obs[watch:watch+3].copy() if watch is not None else None
        for k in range(n+1):
            p=p0+(p1-p0)*k/n
            q=self.ik(p,psi,q)
            a=np.zeros(11); a[3:10]=4*(q+bias-self.QT); a[10]=self.grip
            self.step(a); self.step(np.r_[np.zeros(10),self.grip])
            qe=np.abs(self.obs[96:103]+bias-self.QT).max()
            if ref is not None and np.linalg.norm(self.obs[watch:watch+3]-ref)>thr:
                return p,'obj',qe
            if qe>qthr: return p,'q',qe
        return p,'none',qe
