import numpy as np
from env_client import make_env
for seed in [0,1,2,5]:
 e=make_env();o,_=e.reset(seed=seed);q=o[128:135].copy();q[1]=-1.15;q[3]=-3
 for k in range(20):
  a=np.zeros(11,np.float32);a[3:10]=np.clip((q-o[128:135])*.6,-.1,.1);a[10]=1;o,r,t,tr,i=e.step(a)
 print(seed,'base',o[125:128],'drawer',np.round(o[103:109],3),'wiper',np.round(o[147:150],3));e.close()
