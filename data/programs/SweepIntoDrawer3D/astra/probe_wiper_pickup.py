import numpy as np
from env_client import make_env
from kinova import fk,ik
np.set_printoptions(precision=4,suppress=True)
e=make_env()
for dx in [.08,0,.14]:
 for z in [.04,.06,.08]:
  s,_=e.reset(seed=0);h=s[128:135].copy();rot=fk(h)[:3,:3];hi=ik([.5,0,.2],rot,h,max_nfev=100);lo=ik([.5,0,z],rot,hi,max_nfev=100);w=s[147:150].copy();xy=w[:2]+[.7+dx,0]
  def move(q,g,n):
   global s
   for k in range(n):
    a=np.zeros(11,np.float32);a[:2]=np.clip(1.2*(xy-s[125:127]),-.05,.05);a[2]=np.clip(np.pi-s[127],-.1,.1);a[3:10]=np.clip(2*(q-s[128:135]),-.1,.1);a[10]=g;s,*_=e.step(a)
  move(hi,0,70);move(lo,0,35);pre=s[147:150].copy();f=fk(s[128:135])[:3,3].copy();move(lo,1,5);closed=s[147:150].copy();move(hi,1,40)
  print('trial',dx,z,'fk_low',f,'base',s[125:128],'pre',pre-w,'close',closed-pre,'lift',s[147:150]-closed,'qerr',max(abs(s[128:135]-hi)),flush=True)
  np.save('pickup_'+str(dx)+'_'+str(z)+'.npy',s)
  if s[149]>.49:
   print('PICKUP',dx,z,flush=True);xy+=np.array([.2,0]);move(hi,1,15);print('PULL',s[147:150]-w,flush=True);np.save('pickup_success.npy',s)
e.close()
