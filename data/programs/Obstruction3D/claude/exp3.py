import numpy as np
from env_client import make_env
import fk, ik
from servo import *

env = make_env()
obs, info = env.reset(seed=0)
q = robot_q(obs); b = robot_base(obs)
blk = opos(obs,'target_block')
print("block", blk, "base", b)
target_xy = np.array([blk[0]-b[0], blk[1]-b[1]])
Z = 0.30
Rd = down_R(0.0)
steps=0
def go(p_des, Rd, obs, q, nmax=60):
    global steps
    last=None
    for i in range(nmax):
        dq,ep,ew = ik.ik_step(q,p_des,Rd,tool_z=0.0,damp=0.03,max_delta=0.15)
        a=np.zeros(11,dtype=np.float32); a[3:10]=dq
        obs,rew,t,tr,inf = env.step(a); steps+=1
        qn = robot_q(obs)
        if np.max(np.abs(qn-(q+dq)))>1e-5:
            return obs,qn,ep,ew,True
        q=qn
        if ep<1e-3 and ew<5e-3: break
    return obs,q,ep,ew,False

p = np.array([target_xy[0], target_xy[1], Z])
obs,q,ep,ew,blocked = go(p,Rd,obs,q)
print("above:",np.round(fk.fk_ee(q)[:3,3],4),"ep",round(ep,4),"ew",round(ew,4),"blocked",blocked,"steps",steps)
# descend
for Z in np.arange(0.28,0.05,-0.01):
    p = np.array([target_xy[0], target_xy[1], Z])
    obs,q,ep,ew,blocked = go(p,Rd,obs,q,nmax=25)
    act = fk.fk_ee(q)[:3,3]
    print("Z",round(Z,3),"actual",np.round(act,4),"blocked",blocked, rinfo(obs))
    if blocked: break
print("steps",steps)
