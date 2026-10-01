import numpy as np
from env_client import make_env
for tx in [.78,.84,.90,.96,1.02]:
 for yaw in [1.40,1.57,1.74]:
  e=make_env();o,_=e.reset(seed=3)
  def go(t,n):
   global o
   for _ in range(n):
    a=np.zeros(11,np.float32);a[:3]=np.clip((np.array(t)-o[125:128])*.8,-.1,.1);a[10]=1;o,*_=e.step(a)
  go([o[125],-1.35,o[127]],20);go([tx,-1.35,yaw],28);go([tx,-.92,yaw],15)
  print(tx,yaw,np.round(o[103:109],3),np.round(o[125:128],2));e.close()
