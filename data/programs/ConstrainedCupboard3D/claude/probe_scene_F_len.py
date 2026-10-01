import numpy as np
from env_client import make_env
env = make_env(); obs, info = env.reset(seed=0, options={'object_count':3})
a=np.zeros(11,dtype=np.float32); tot=0.0
for i in range(1500):
    obs,r,term,trunc,info=env.step(a); tot+=r
    if term or trunc:
        print("END at step",i+1,"term",term,"trunc",trunc,"r",r,"total",tot,"info",info); break
else:
    print("no end in 1500 steps, total",tot)
env.close()
