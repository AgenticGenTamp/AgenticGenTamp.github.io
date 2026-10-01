import time, numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=1)
R=lambda f: float(obs.get(obs.get_object_from_name("robot"),f))
def J(): return np.array([R("pos_arm_joint%d"%i) for i in range(1,8)])
def step(a):
    global obs
    obs,rew,term,trunc,info=env.step(np.asarray(a,dtype=np.float32)); return rew
try:
    s=env.get_state(); print("get_state OK len", len(np.asarray(s).ravel()))
except Exception as e:
    print("get_state ERR", repr(e)[:120])
def goto(tgt,n=35,grip=0.0):
    tgt=np.asarray(tgt,float)
    for i in range(n):
        a=np.zeros(11); a[3:10]=np.clip(tgt-J(),-0.1,0.1); a[10]=grip
        step(a)
    e=J()-tgt
    print(" tgt",np.round(tgt,3).tolist())
    print(" got",np.round(J(),3).tolist(),"err",np.round(e,3).tolist(),"maxerr",round(np.abs(e).max(),3))
cands={
 "A_home":[0,0.262,3.142,-2.269,0,0.960,1.571],
 "B_fwd_down":[0,0.6,3.142,-1.6,0,-1.2,1.571],
 "C_straight_down":[0,0.9,3.142,-0.9,0,-1.4,1.571],
 "D_deep":[0,1.3,3.142,-0.6,0,-1.6,1.571],
}
for k,v in cands.items():
    print("=== ",k); t=time.time(); goto(v); print(" %.1fs"%(time.time()-t))
env.close()
