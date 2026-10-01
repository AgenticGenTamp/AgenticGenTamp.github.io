import numpy as np
from env_client import make_env
env=make_env(); obs,_=env.reset(seed=0); tot=0
for k in range(1200):
    obs,r,t,tr,i=env.step(np.zeros(11)); tot+=r
    if t or tr:
        print("ended at step",k+1,"term",t,"trunc",tr,"info",i,"total",round(tot,3)); break
else: print("no end in 1200, total",round(tot,3))
env.close()
