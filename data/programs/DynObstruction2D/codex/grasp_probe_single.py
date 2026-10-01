import numpy as np
from env_client import make_env
e=make_env(); s,_=e.reset(seed=0)
for i,a in enumerate(([.049,0,0,0,0],[0,.049,0,0,0],[-.049,0,0,0,0],[0,-.049,0,0,0])):
    try:
        s,r,t,tr,info=e.step(np.array(a,dtype=np.float32)); print(i,"ok")
    except Exception as x: print(i,repr(x));break
e.close()
