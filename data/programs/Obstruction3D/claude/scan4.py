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
xs=[float(v) for v in sys.argv[1].split(',')]
close_first = len(sys.argv)>2 and sys.argv[2]=='cf'
env=make_env(); obs,info=env.reset(seed=0); q=robot_q(obs); b=robot_base(obs)
blk=opos(obs,'target_block'); Rd=down_R(0.0)
for dx in xs:
    for dy in [-0.02,0.0,0.02]:
        p=np.array([blk[0]-b[0]+dx, blk[1]-b[1]+dy, 0.33])
        obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=40)
        if close_first:
            a=np.zeros(11,dtype=np.float32); a[10]=-1.0; obs,r,t,tr,inf=env.step(a)
        # descend in 5mm steps, closing at each
        for Z in np.arange(0.32,0.215,-0.005):
            p[2]=Z
            obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=10)
            a=np.zeros(11,dtype=np.float32); a[10]=-1.0
            obs,r,t,tr,inf=env.step(a)
            ri=rinfo(obs)
            if ri['grasp_active']>0.5:
                print("GRASP dx",round(dx,3),"dy",dy,"Z",round(Z,3),ri,flush=True); break
            a[10]=1.0; obs,r,t,tr,inf=env.step(a)
            if bl: break
        p[2]=0.33; obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=30)
    print("dx",round(dx,3),"done",flush=True)
