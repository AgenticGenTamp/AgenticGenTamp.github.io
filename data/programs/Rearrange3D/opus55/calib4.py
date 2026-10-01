import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env(); obs,_=env.reset(seed=0)
for j in [0,6]:
  for v in [0.1,0.04,0.02,0.01,-0.01]:
    q0=obs[96:103].copy()
    for k in range(10):
        a=np.zeros(11,np.float32); a[3+j]=v; obs,*_=env.step(a)
    q1=obs[96:103].copy()
    for k in range(10):
        a=np.zeros(11,np.float32); obs,*_=env.step(a)
    print(j,v,'moved during',(q1-q0)[j],'after settle',(obs[96:103]-q0)[j])
