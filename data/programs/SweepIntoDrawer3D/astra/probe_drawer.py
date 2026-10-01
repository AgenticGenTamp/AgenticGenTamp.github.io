from env_client import make_env
import numpy as np
import json
np.set_printoptions(precision=3,suppress=True)
e=make_env()
for dj in [0,.2,.4,-.2,-.4]:
 for dx in [.22,.32,.42,.52]:
  s,_=e.reset(seed=0);q=s[128:135].copy();q[1]+=dj
  target=np.array([1.02+dx,0.,np.pi])
  for t in range(25):
   a=np.zeros(11);a[:3]=np.clip(target-s[125:128],-.1,.1);a[3:10]=np.clip(q-s[128:135],-.1,.1);s,*_=e.step(a)
  for t in range(3):
   a=np.zeros(11);a[-1]=1;s,*_=e.step(a)
  for t in range(8):
   a=np.zeros(11);a[0]=.07;a[-1]=1;s,r,d,tr,i=e.step(a)
  print(dj,dx,'draw',s[103:109].round(3).tolist(),flush=True)
  if max(s[103:109])>.02:
   np.save('drawer_success.npy',s);print('SUCCESS',flush=True)
e.close()
