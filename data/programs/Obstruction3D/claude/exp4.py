import numpy as np
from env_client import make_env
import fk, ik
from servo import *

def make(seed=0):
    env = make_env(); obs,info = env.reset(seed=seed); return env,obs

def go(env,obs,q,p_des,Rd,nmax=60,tol=1e-3):
    for i in range(nmax):
        dq,ep,ew = ik.ik_step(q,p_des,Rd,tool_z=0.0,damp=0.03,max_delta=0.15)
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq
        obs,rew,t,tr,inf = env.step(a)
        qn = robot_q(obs)
        if np.max(np.abs(qn-(q+dq)))>1e-5:
            return obs,qn,ep,ew,True
        q=qn
        if ep<tol and ew<5e-3: break
    return obs,q,ep,ew,False

# 1. descend over free table spot
env,obs = make(0); q=robot_q(obs); b=robot_base(obs)
Rd = down_R(0.0)
free = np.array([0.30, -0.25])  # world, empty
p=np.array([free[0]-b[0], free[1]-b[1], 0.30])
obs,q,ep,ew,bl = go(env,obs,q,p,Rd)
print("free above ok",bl,ep)
zb=None
for Z in np.arange(0.29,0.02,-0.005):
    p=np.array([free[0]-b[0], free[1]-b[1], Z])
    obs,q,ep,ew,bl = go(env,obs,q,p,Rd,nmax=20)
    if bl:
        zb=Z; break
print("blocked over free table at Z=",zb, "actual", np.round(fk.fk_ee(q)[:3,3],4))
env.close()

# 2. approach block, close gripper at various heights
for Zg in [0.26,0.24,0.23,0.235]:
    env,obs = make(0); q=robot_q(obs); b=robot_base(obs)
    blk = opos(obs,'target_block')
    p=np.array([blk[0]-b[0], blk[1]-b[1], 0.30])
    obs,q,ep,ew,bl = go(env,obs,q,p,Rd)
    p[2]=Zg
    obs,q,ep,ew,bl = go(env,obs,q,p,Rd,nmax=30)
    a=np.zeros(11,dtype=np.float32); a[10]=-1.0
    obs,rew,t,tr,inf=env.step(a)
    print("Zg",Zg,"reach_blocked",bl,"ep",round(ep,4),rinfo(obs))
    # try lifting
    p[2]=0.30
    obs,q,ep,ew,bl = go(env,obs,q,p,Rd,nmax=30)
    print("   after lift block pos", np.round(opos(obs,'target_block'),4), rinfo(obs))
    env.close()
