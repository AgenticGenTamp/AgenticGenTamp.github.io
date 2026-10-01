from env_client import make_env
import numpy as np
np.set_printoptions(precision=4,suppress=True)
e=make_env()
for dj in [0,-.3,.3,.6]:
 s,_=e.reset(seed=0);q=s[128:135].copy();q[3]+=dj
 for bx in [1.15,1.0,1.3,1.45]:
  for by in np.arange(-.7,.71,.1):
   for k in range(12):
    a=np.zeros(11);a[:3]=np.clip([bx,by,np.pi]-s[125:128],-.1,.1);a[3:10]=np.clip(q-s[128:135],-.1,.1);s,*_=e.step(a)
   before=s[103:109].copy()
   for k in range(4):
    a=np.zeros(11);a[10]=1;a[0]=.05 if k else 0;s,r,d,tr,i=e.step(a)
   change=s[103:109]-before
   if max(change)>.015:
    print('CONTACT',dj,bx,by,change,'statebase',s[125:128],flush=True)
    np.save('drawer_scan_contact.npy',s)
   if max(s[103:109])>.1:print('OPEN',s[103:109],flush=True)
  print('row',dj,bx,'draw',s[103:109],flush=True)
e.close()
