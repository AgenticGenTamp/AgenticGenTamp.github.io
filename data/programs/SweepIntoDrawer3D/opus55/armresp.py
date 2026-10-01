import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0)
q0=obs[128:135].copy()
# constant delta 0.05 on joint 1 for 10 steps then zero
for t in range(14):
    a=np.zeros(11); 
    if t<8: a[4]=0.05
    obs,*_=env.step(a); print(t, (obs[128:135]-q0).round(4)[:3])
q0=obs[128:135].copy(); tgt=q0[1]-0.3
print("P control K=1")
for t in range(14):
    a=np.zeros(11); a[4]=np.clip(tgt-obs[129],-0.1,0.1)
    obs,*_=env.step(a); print(t, round(obs[129]-tgt,4))
env.close()
