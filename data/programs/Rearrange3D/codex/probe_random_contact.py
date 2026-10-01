from env_client import make_env
import numpy as np

rng=np.random.default_rng(3)
env=make_env(); o,_=env.reset(seed=0)
prev=o.copy(); initial=o.copy()
for k in range(160):
    # explore smoothly, maintaining each random velocity 4 steps
    if k%4==0:
        a=np.zeros(11,np.float32)
        a[3:10]=rng.choice([-0.1,0.1],7)
        a[10]=rng.integers(0,2)
    old=o.copy(); o,r,t,tr,_=env.step(a)
    delta=o[[0,1,2,16,17,18,32,33,34]]-old[[0,1,2,16,17,18,32,33,34]]
    total=o[[0,1,2,16,17,18,32,33,34]]-initial[[0,1,2,16,17,18,32,33,34]]
    if np.max(np.abs(delta))>.004 or r!=-1:
        print(k,'r',r,'a',a.tolist(),'delta',np.round(delta,3).tolist(),'total',np.round(total,3).tolist(),'j',np.round(o[96:104],2).tolist(),flush=True)
    if t or tr: print('done',k,t,tr); break
env.close()
