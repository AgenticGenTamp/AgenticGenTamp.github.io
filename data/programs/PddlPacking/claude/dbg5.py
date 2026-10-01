import numpy as np
from env_client import make_env
from approach import GeneratedApproach, Robot, rotz, quat_R, BLOCK_ORIENTS, wrap_joints
import fk
env=make_env(); obs,info=env.reset(seed=7)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for n in range(63):
    a=ap.get_action(obs); obs,r,t,tr,i=env.step(a)
R=Robot(obs)
def cart(pw,Rd,maxit=40,step=0.06):
    global obs,R
    for i in range(maxit):
        dq,conv=ap.cart_step(R,pw,Rd,step_len=step)
        if conv: return True,i
        if dq is None: return "ikstuck",i
        prev=R.q.copy()
        obs,r,t,tr,inf=env.step(ap.act_arm(dq)); R=Robot(obs)
        if np.allclose(prev,R.q): return "rej",i
    return "maxit",i
p,Rw,_=R.tool()
print("start",np.round(p,3))
for z in [1.05,1.15]:
    print("up to",z,cart(np.array([p[0],p[1],z]),Rw))
    p2,Rw2,_=R.tool(); print("  tool",np.round(p2,3))
    b=R.blocks["block2"]
    Rrel=Rw2.T@quat_R(b[3:7])
    best,bd=None,1e9
    for Rb in BLOCK_ORIENTS:
        Rd=Rb@Rrel.T; d=np.linalg.norm(fk.so3_error(Rw2,Rd))
        if d<bd: bd,best=d,Rd
    off=best@(Rw2.T@(np.array(b[:3])-p2))
    tgt=np.array([0-off[0],0.088-off[1],z])
    print("  to cell",cart(tgt,best),np.round(R.tool()[0],3),"blk",np.round(R.blocks["block2"][:3],3))
env.close()
