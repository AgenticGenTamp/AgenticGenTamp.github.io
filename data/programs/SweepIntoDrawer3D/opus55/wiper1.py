import numpy as np, sys
from env_client import make_env
from kin import *
np.set_printoptions(precision=4, suppress=True, linewidth=150)
env = make_env()
def goto(qd, n=80, grip=0.0, tol=1e-3):
    global obs
    for t in range(n):
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-obs[128:135],-0.1,0.1); a[10]=grip
        obs,*_=env.step(a)
        if np.max(np.abs(obs[128:135]-qd))<tol: break
    return obs[128:135]
for yaw in [0.0, np.pi/2]:
  for dx,dy in [(0,0)]:
    obs,_=env.reset(seed=0)
    base=obs[125:128].copy(); qc=obs[128:135].copy()
    w=obs[147:150].copy()
    for z,g in [(0.56,0),(0.47,0),(0.455,0),(0.455,1),(0.455,1),(0.56,1)]:
        qd,err=ik(base,qc,np.array([w[0]+dx,w[1]+dy,z]),R_down(yaw)); goto(qd,grip=g); qc=qd
        T=fk_world(base,obs[128:135])
        print(yaw, z,g,"tool",T[:3,3],"wiper",obs[147:154])
env.close()
