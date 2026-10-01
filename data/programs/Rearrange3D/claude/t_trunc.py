import numpy as np, time
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0)
z=np.zeros(11,dtype=np.float32)
t0=time.time(); n=0
while True:
    obs,r,te,tr,info=env.step(z); n+=1
    if te or tr or n>=1100: break
print("steps", n, "term", te, "trunc", tr, "elapsed", round(time.time()-t0,2), "steps/s", round(n/(time.time()-t0),2))
env.close()
