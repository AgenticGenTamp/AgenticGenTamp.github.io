from env_client import make_env
from kinova import planar_ik
import numpy as np
np.set_printoptions(precision=3,suppress=True)
e=make_env()
for z in [-.05,-.1,0.,-.15]:
 s,_=e.reset(seed=0);q=planar_ik(.6,z,pitch=-np.pi/2)
 def move(x,n,g):
  global s
  for k in range(n):
   a=np.zeros(11);a[:3]=np.clip([x,0,np.pi]-s[125:128],-.02,.02);a[3:10]=np.clip(2*(q-s[128:135]),-.1,.1);a[10]=g;s,r,d,tr,i=e.step(a)
 move(2.1,160,0)
 print('pose',z,q,'err',max(abs(q-s[128:135])),flush=True)
 for bx in [1.85,1.8,1.75,1.7,1.65]:
  move(bx,35,0);move(bx,5,1);move(2.1,35,1)
  print(z,bx,'draw',s[103:109].round(3),'qerr',round(max(abs(q-s[128:135])),3),flush=True)
  if max(s[103:109])>.12:np.save('horizontal_drawer_success.npy',s);break
e.close()
