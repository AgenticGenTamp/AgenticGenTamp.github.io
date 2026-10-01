import numpy as np, time
from env_client import make_env
from kin import *
env=make_env(); o,_=env.reset(seed=0)
t=time.time()
for i in range(100): o,*_=env.step(np.zeros(11,np.float32))
print("step ms", (time.time()-t)*10)
t=time.time()
for i in range(20): ik(o[125:128],o[128:135],np.array([0.8,0,0.5]),R_down(0),iters=150)
print("ik ms", (time.time()-t)*50)
