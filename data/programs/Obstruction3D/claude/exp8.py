import numpy as np
from env_client import make_env
import fk, ik
from servo import *
def go(env,obs,q,p_des,Rd,nmax=40,tol=1e-3):
    for i in range(nmax):
        dq,ep,ew = ik.ik_step(q,p_des,Rd,tool_z=0.0,damp=0.03,max_delta=0.15)
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq
        obs,rew,t,tr,inf = env.step(a)
        qn = robot_q(obs)
        if np.max(np.abs(qn-(q+dq)))>1e-5: return obs,qn,ep,ew,True
        q=qn
        if ep<tol and ew<5e-3: break
    return obs,q,ep,ew,False
env=make_env(); obs,inf=env.reset(seed=1); q=robot_q(obs); b=robot_base(obs)
Rd=down_R(0.0)
# map table: for a grid of world x,y descend and find lowest reachable Z
print("obstruction",opos(obs,'obstruction0'),"tb",opos(obs,'target_block'),"tr",opos(obs,'target_region'))
for x in [0.0,0.05,0.1,0.2,0.3,0.4,0.5,0.55]:
    row=[]
    for y in [-0.5,-0.3,0.0,0.3,0.5]:
        p=np.array([x-b[0],y-b[1],0.35])
        obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=40)
        if bl or ep>0.01:
            row.append((y,'unreach')); continue
        zl=None
        for Z in np.arange(0.34,0.02,-0.02):
            p[2]=Z
            obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=15)
            if bl: zl=Z; break
        row.append((y, 'free' if zl is None else round(zl,2)))
        p[2]=0.35; obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=30)
    print("x",x,row, flush=True)
