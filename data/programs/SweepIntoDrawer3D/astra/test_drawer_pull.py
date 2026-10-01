from env_client import make_env
import numpy as np
for y in [-.3,.3,-.6,0]:
 e=make_env();s,_=e.reset(seed=0);q=s[128:135].copy()
 for stage,(bx,by,n,g) in enumerate([(1.4,y,20,0),(1.15,y,25,0),(1.15,y,5,1),(1.7,y,20,1)]):
  for k in range(n):
   a=np.zeros(11);a[:3]=np.clip([bx,by,np.pi]-s[125:128],-.05,.05);a[3:10]=np.clip(q-s[128:135],-.1,.1);a[10]=g;s,r,d,tr,i=e.step(a)
  print(y,stage,'base',s[125:128].round(3),'draw',s[103:109].round(3),flush=True)
 np.save('pull_%s.npy'%y,s);e.close()
