import numpy as np
from env_client import make_env
np.set_printoptions(precision=3, suppress=True, linewidth=250)
def run(deltas,steps=40,seed=0,pre=None):
    env=make_env(); obs,_=env.reset(seed=seed)
    if pre is not None:
        for k in range(pre[1]):
            obs,r,t,tr,_=env.step(pre[0])
    c0=obs[:80].reshape(5,16)[:,:3].copy(); w0=obs[147:150].copy()
    for k in range(steps):
        obs,r,t,tr,_=env.step(deltas)
    c=obs[:80].reshape(5,16)[:,:3]
    env.close()
    return obs,r,np.linalg.norm(c-c0,axis=1),np.linalg.norm(obs[147:150]-w0)
env=make_env(); obs,_=env.reset(seed=0); print("q0",np.round(obs[128:136],3)); env.close()
for j in range(7):
    for s in (0.1,-0.1):
        a=np.zeros(11); a[3+j]=s
        o,r,dc,dw=run(a)
        print("joint",j,s,"q",np.round(o[128:136],2),"cubemoved",np.round(dc,3),"wiper",round(float(dw),3),"r",r)
