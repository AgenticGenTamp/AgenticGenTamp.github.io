import numpy as np
from env_client import make_env
env=make_env()
for rot in [0,0.4,0.8,1.2,1.57,2.0,2.4,3.14,-1.57,-0.8]:
    o,i=env.reset(seed=72)
    a=np.zeros(11); a[0]=1.35/8; 
    for t in range(8): o,*_=env.step(a)
    a=np.zeros(11); a[2]=rot/8
    for t in range(8): o,*_=env.step(a)
    a=np.zeros(11); a[1]=-0.02
    for t in range(120): o,r,te,*_=env.step(a)
    print(rot,o[:3],te)
