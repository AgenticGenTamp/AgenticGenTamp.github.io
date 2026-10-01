import numpy as np
from probe_vis_lib import *
env,obs,info=new_env(1)
O=feats(obs,'objective0'); op=np.array([O['x'],O['y']])
start=pose(obs,0)[:2]; u=(start-op); u/=np.linalg.norm(u)
def trials(obs,d,n=8):
    tgt=op+u*d
    obs,ok=goto(env,obs,tgt[0],tgt[1],i=0,tol=0.001)
    dd=np.linalg.norm(pose(obs,0)[:2]-op)
    cnt=0
    for _ in range(n):
        if rf(obs,0)['calibrated']>0.5: obs,_,_,_,_=st(env,op='image',i=0)
        obs,_,_,_,_=st(env,op='calibrate',i=0)
        cnt+= rf(obs,0)['calibrated']>0.5
    if rf(obs,0)['calibrated']>0.5: obs,_,_,_,_=st(env,op='image',i=0)
    print("d=%.4f  successes %d/%d"%(dd,cnt,n))
    return obs
for d in [1.5,1.9,1.99,2.01,2.03,2.05,2.1,2.2,2.5]:
    obs=trials(obs,d)
env.close()
