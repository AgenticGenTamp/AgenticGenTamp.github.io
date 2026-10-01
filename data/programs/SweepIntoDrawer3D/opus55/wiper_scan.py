import numpy as np, sys
from env_client import make_env
from kin import *
from concurrent.futures import ThreadPoolExecutor
np.set_printoptions(precision=3, suppress=True, linewidth=200)
xs=np.arange(-0.15,0.151,0.025); ys=np.arange(-0.15,0.151,0.025)
def row(ix):
    env=make_env(); obs,_=env.reset(seed=0)
    base=obs[125:128].copy(); qc=obs[128:135].copy(); w0=obs[147:150].copy()
    out=[]
    def goto(qd,n=40):
        nonlocal obs
        for t in range(n):
            a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-obs[128:135],-0.1,0.1); a[10]=1
            obs,*_=env.step(a)
            if np.max(np.abs(obs[128:135]-qd))<1e-3: break
    for dy in ys:
        p=w0+np.array([xs[ix],dy,0])
        qd,_=ik(base,qc,np.array([p[0],p[1],0.50]),R_down(np.pi/2)); goto(qd); qc=qd
        qd2,_=ik(base,qc,np.array([p[0],p[1],0.43]),R_down(np.pi/2)); goto(qd2,25)
        out.append(fk_world(base,obs[128:135])[2,3]); 
        goto(qd)
    moved=np.linalg.norm(obs[147:150]-w0)
    env.close(); return out, moved
with ThreadPoolExecutor(13) as ex: res=list(ex.map(row,range(len(xs))))
print("rows=dx, cols=dy", ys)
for ix,(o,m) in enumerate(res): print(f"{xs[ix]:+.3f}", np.array(o), f"moved={m:.3f}")
