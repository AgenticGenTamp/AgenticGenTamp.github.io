import sys, numpy as np
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
yaw=float(sys.argv[1]); axis=sys.argv[2]; fixed=float(sys.argv[3]); seed=int(sys.argv[4]); target=sys.argv[5]
env=make_env(); obs,info=env.reset(seed=seed); q=robot_q(obs); b=robot_base(obs)
blk=opos(obs,target); Rd=down_R(yaw)
vals=np.arange(-0.10,0.101,0.01)
out=[]
for v in vals:
    dx,dy = (v,fixed) if axis=='x' else (fixed,v)
    p=np.array([blk[0]-b[0]+dx, blk[1]-b[1]+dy, 0.33])
    obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=40)
    if bl or ep>0.005:
        out.append((round(v,3),'X')); continue
    zl=0.0
    for Z in np.arange(0.32,0.21,-0.01):
        p[2]=Z
        obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=12)
        if bl: zl=Z+0.01; break
    out.append((round(v,3),round(zl,2)))
    p[2]=0.33; obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=25)
print("yaw",yaw,"axis",axis,"fixed",fixed,"seed",seed,target,out,flush=True)
