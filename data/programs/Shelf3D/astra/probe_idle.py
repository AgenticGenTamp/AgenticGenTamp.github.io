from env_client import make_env
import numpy as np,time
e=make_env();s,i=e.reset(seed=1);start=time.monotonic();a=np.zeros(11);last=None
for j in range(1001):
    s,r,t,tr,i=e.step(a)
    if (r,t,tr)!=last or j%100==0: print(j+1,r,t,tr,i,'sec',time.monotonic()-start,flush=True)
    last=(r,t,tr)
    if t or tr or time.monotonic()-start>55:break
e.close()
