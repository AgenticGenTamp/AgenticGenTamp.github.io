from env_client import make_env
import numpy as np
np.set_printoptions(precision=4,suppress=True)
e=make_env(); s,info=e.reset(seed=0)
print('INFO',info,'max',e.max_steps); print('INITIAL',s)
for k in range(11):
 s,_=e.reset(seed=0); a=np.zeros(11); a[k]=.1 if k<10 else 1
 n,r,t,tr,i=e.step(a); print('ACTION',k,'delta',n-s,'reward',r,t,tr,i)
e.close()
