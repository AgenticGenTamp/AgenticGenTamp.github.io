import numpy as np, sys
from env_client import make_env
from kin import *
np.set_printoptions(precision=3, suppress=True, linewidth=150)
env = make_env()
obs,_=env.reset(seed=0); steps=0; rew=[]
def goto(qd, n=80, grip=0.0, tol=0.01):
    global obs, steps
    for t in range(n):
        a=np.zeros(11,dtype=np.float32); a[3:10]=np.clip(qd-obs[128:135],-0.1,0.1); a[10]=grip
        obs,r,*_=env.step(a); steps+=1; rew.append(r)
        if np.max(np.abs(obs[128:135]-qd))<tol: break
base=obs[125:128].copy(); qc=obs[128:135].copy()
ci=0; c=obs[16*ci:16*ci+3]; qw,qz=obs[16*ci+3],obs[16*ci+6]
yaw=2*np.arctan2(qz,qw); yaw=(yaw+np.pi/4)%(np.pi/2)-np.pi/4
for p,g,tol in [((c[0],c[1],0.52),0,0.02),((c[0],c[1],0.445),0,0.005),(None,1,0),((c[0],c[1],0.53),1,0.02),((0.98,c[1],0.53),1,0.02),(None,0,0)]:
    if p is None:
        for _ in range(4): goto(qc,1,grip=g)
    else:
        qd,err=ik(base,qc,np.array(p),R_down(yaw)); goto(qd,grip=g,tol=tol); qc=qd
    print(steps, fk_world(base,obs[128:135])[:3,3], obs[16*ci:16*ci+3])
for _ in range(20): goto(qc,1)
print(steps, obs[16*ci:16*ci+3], set(rew))
env.close()
