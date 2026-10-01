import numpy as np
from env_client import make_env
from probe_dynamics import vals
e=make_env();s,_=e.reset(seed=42)
a=np.zeros(11);a[3]=.1
for j in range(500):
 s,r,t,tr,i=e.step(a)
 if (j+1)%100==0: print(j+1,vals(s)['robot'],flush=True)
e.close()
