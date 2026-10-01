import numpy as np
from env_client import make_env
from kin import fk_world, ik, R_down
np.set_printoptions(precision=3,suppress=True,linewidth=150)
# tool pointing -x, tool x axis = world -z (variant A) ; variant B: tool x = world y
RA = np.array([[0,0,-1],[0,-1,0],[-1,0,0]],float).T  # columns x,y,z
RA = np.stack([[0,0,-1],[0,-1,0],[-1,0,0]],axis=1).astype(float)
RB = np.stack([[0,1,0],[0,0,-1],[-1,0,0]],axis=1).astype(float)
class R:
    def __init__(s,seed=0):
        s.env=make_env(); s.obs,_=s.env.reset(seed=seed); s.rews=[]; s.grip=0.0; s.n=0
    @property
    def base(s): return s.obs[125:128].copy()
    @property
    def q(s): return s.obs[128:135].copy()
    def step(s,a):
        s.obs,r,t,tr,info=s.env.step(np.asarray(a,np.float32)); s.rews.append(r); s.n+=1; return r
    def goto(s,qd,n=60,base_d=None,tol=0.003):
        for i in range(n):
            a=np.zeros(11); a[3:10]=np.clip(qd-s.q,-0.1,0.1); a[10]=s.grip
            if base_d is not None:
                d=base_d-s.base; a[0:3]=np.clip(d,-0.1,0.1)
            s.step(a)
            if np.max(np.abs(qd-s.q))<tol and (base_d is None or np.max(np.abs(base_d-s.base))<0.003): break
        return np.max(np.abs(qd-s.q))
    def tool(s): return fk_world(s.base,s.q)
    def ik(s,p,Rd,q0=None):
        q,e=ik(s.base,s.q if q0 is None else q0,np.asarray(p,float),Rd); return q,e
    def hold(s,n=10):
        for i in range(n):
            a=np.zeros(11); a[10]=s.grip; s.step(a)
