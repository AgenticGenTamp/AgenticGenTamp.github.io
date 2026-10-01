import numpy as np
from env_client import make_env
from kinova import fk
np.set_printoptions(precision=3,suppress=True)
e=make_env()
for offset in [.1,.2,.3,.4,.5]:
 for close in [1,0]:
  s,_=e.reset(seed=0);q=s[128:135].copy();q[3]+=offset;w=s[147:150].copy();f=fk(q)
  xy=np.array([w[0]+.2+f[0,3],w[1]+f[1,3]])
  def move(xy,g,n):
   global s
   for k in range(n):
    a=np.zeros(11,np.float32);a[:2]=np.clip(1.2*(xy-s[125:127]),-.1,.1);a[2]=np.clip(np.pi-s[127],-.1,.1);a[3:10]=np.clip(2.5*(q-s[128:135]),-.1,.1);a[10]=g
    s,*_=e.step(a)
  move(xy,1-close,25);pre=s[147:150].copy();np.save('height_before_'+str(offset)+'_'+str(close)+'.npy',s)
  move(xy,close,4);at=s[147:150].copy();move(xy+[.2,0],close,8)
  print('offset',offset,'close',close,'fk',f[:3,3],'qerr',np.max(abs(s[128:135]-q)),'pre',pre-w,'at',at-pre,'pull',s[147:150]-at,'drawers',s[103:109],flush=True)
  if np.linalg.norm(s[147:150]-at)>.025:np.save('height_success_'+str(offset)+'_'+str(close)+'.npy',s)
e.close()
