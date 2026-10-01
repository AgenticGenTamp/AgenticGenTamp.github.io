import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0)
for K in (1,2,3,4):
    q0=obs[128:135].copy(); tgt=q0.copy(); tgt[1]-=0.3; tgt[3]+=0.3; tgt[5]+=0.2
    errs=[]
    for t in range(22):
        a=np.zeros(11); a[3:10]=np.clip(K*(tgt-obs[128:135]),-0.1,0.1)
        obs,*_=env.step(a); errs.append(np.abs(obs[128:135]-tgt).max()*np.sign((obs[129]-tgt[1])*-1))
    print(K, np.round(errs,3))
    # return
    for t in range(15):
        a=np.zeros(11); a[3:10]=np.clip(2*(q0-obs[128:135]),-0.1,0.1); obs,*_=env.step(a)
env.close()
