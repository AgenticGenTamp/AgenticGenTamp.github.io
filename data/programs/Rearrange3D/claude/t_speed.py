import numpy as np, time
from env_client import make_env
env=make_env(); env.reset(seed=0)
z=np.zeros(11,dtype=np.float32)
a=np.zeros(11,dtype=np.float32); a[3:10]=0.1; a[0]=0.1
for lbl,act in [("zero",z),("full",a)]:
    env.reset(seed=0)
    t0=time.time()
    for i in range(200): env.step(act)
    dt=time.time()-t0
    print(lbl, "200 steps", round(dt,2),"s ->", round(200/dt,1),"steps/s")
env.close()
