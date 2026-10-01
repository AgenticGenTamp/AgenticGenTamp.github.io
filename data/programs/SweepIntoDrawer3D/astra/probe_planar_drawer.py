from env_client import make_env
from kinova import planar_ik
import numpy as np
np.set_printoptions(precision=3,suppress=True)
e=make_env()
for z in [-.05,0.,-.1,.05]:
 for bx in [1.65,1.6,1.7,1.55]:
  s,_=e.reset(seed=0);q=planar_ik(.5,z)
  for stage,(x,n,g) in enumerate([(1.9,100,0),(bx,30,0),(bx,4,1),(2.,25,1)]):
   for k in range(n):
    a=np.zeros(11);a[:3]=np.clip([x,0,np.pi]-s[125:128],-.025,.025);a[3:10]=np.clip(q-s[128:135],-.1,.1);a[10]=g;s,r,d,tr,i=e.step(a)
  print(z,bx,'draw',s[103:109].round(3),'qerr',round(max(abs(q-s[128:135])),3),flush=True)
  if max(s[103:109])>.1:np.save('planar_drawer_success.npy',s)
e.close()
