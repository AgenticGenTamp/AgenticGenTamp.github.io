import numpy as np
from env_client import make_env
env=make_env()
for x in [-3,-2,-1,0, 1.35,2.5]:
    o,i=env.reset(seed=72)
    a=np.zeros(11); a[0]=x/8
    for t in range(8): o,*_=env.step(a)
    a=np.zeros(11); a[1]=-0.05
    for t in range(60): o,r,te,*_=env.step(a)
    print(x,o[:2],te)
