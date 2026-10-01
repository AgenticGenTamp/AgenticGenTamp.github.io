import numpy as np
from env_client import make_env
import fk, ik
from servo import *

def go(env,obs,q,p_des,Rd,nmax=60,tol=1e-3):
    for i in range(nmax):
        dq,ep,ew = ik.ik_step(q,p_des,Rd,tool_z=0.0,damp=0.03,max_delta=0.15)
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq
        obs,rew,t,tr,inf = env.step(a)
        qn = robot_q(obs)
        if np.max(np.abs(qn-(q+dq)))>1e-5: return obs,qn,ep,ew,True
        q=qn
        if ep<tol and ew<5e-3: break
    return obs,q,ep,ew,False

env=make_env(); obs,info=env.reset(seed=0); q=robot_q(obs); b=robot_base(obs)
blk=opos(obs,'target_block')
res={}
for yaw in [0.0, np.pi/2]:
    Rd=down_R(yaw)
    p=np.array([blk[0]-b[0],blk[1]-b[1],0.32])
    obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=80)
    for Z in np.arange(0.31,0.224,-0.005):
        p[2]=Z
        obs,q,ep,ew,bl=go(env,obs,q,p,Rd,nmax=20)
        a=np.zeros(11,dtype=np.float32); a[10]=-1.0
        obs,r,t,tr,inf=env.step(a)
        ri=rinfo(obs)
        print("yaw",round(yaw,2),"Z",round(Z,3),"blocked",bl,"ga",ri['grasp_active'],"fs",ri['finger_state'],"bpos",np.round(opos(obs,'target_block'),3))
        a[10]=1.0; obs,r,t,tr,inf=env.step(a)
        if bl: break
