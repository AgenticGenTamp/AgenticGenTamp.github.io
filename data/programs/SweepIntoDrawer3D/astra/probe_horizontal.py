from env_client import make_env
from kinova import fk,ik
from scipy.spatial.transform import Rotation as R
import numpy as np
np.set_printoptions(precision=3,suppress=True)
e=make_env()
for z in [.2,.25,.3,.15,.35]:
 for bx in [1.55,1.45,1.35]:
  s,_=e.reset(seed=0);q0=s[128:135].copy();rot=R.from_euler('y',-np.pi/2).as_matrix()@fk(q0)[:3,:3];q=ik([.4,0,z],rot,q0)
  for stage,(x,n,g) in enumerate([(1.7,35,0),(bx,20,0),(bx,5,1),(bx+.4,20,1)]):
   for k in range(n):
    a=np.zeros(11);a[:3]=np.clip([x,0,np.pi]-s[125:128],-.04,.04);a[3:10]=np.clip(q-s[128:135],-.1,.1);a[10]=g;s,r,d,tr,i=e.step(a)
  print('z,bx',z,bx,'draw',s[103:109].round(3),'qerr',max(abs(q-s[128:135])),flush=True)
  if max(s[103:109])>.05:np.save('horizontal_contact.npy',s)
e.close()
