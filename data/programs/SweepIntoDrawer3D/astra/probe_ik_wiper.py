import numpy as np
from env_client import make_env
from kinova import fk,ik
np.set_printoptions(precision=4,suppress=True)
e=make_env()
for yo in [0,.1,-.1]:
 for z in [.2,.25,.3,.35,.4]:
  s,_=e.reset(seed=0);home=s[128:135].copy();q=ik([.4,0,z],fk(home)[:3,:3],home,max_nfev=100);w=s[147:150].copy();seen=False
  def step(x,n,g=1):
   global s,seen
   for k in range(n):
    a=np.zeros(11,np.float32);a[0]=np.clip(1.2*(x-s[125]),-.03,.03);a[1]=np.clip(1.2*(w[1]+yo-s[126]),-.06,.06);a[2]=np.clip(np.pi-s[127],-.1,.1);a[3:10]=np.clip(2*(q-s[128:135]),-.1,.1);a[10]=g;s,*_=e.step(a)
    d=s[147:150]-w
    if np.linalg.norm(d)>.003 and not seen:
     seen=True;print('CONTACT',yo,z,'base',s[125:128],'joint',s[128:135],'wdelta',d,flush=True);np.save('ikwiper_contact_'+str(yo)+'_'+str(z)+'.npy',s)
  step(1.8,65,0);step(1.15,30);step(1.8,30)
  print('done',yo,z,'q',q,'fk',fk(q)[:3,3],'base',s[125:128],'delta',s[147:150]-w,flush=True)
  if seen:np.save('ikwiper_end_'+str(yo)+'_'+str(z)+'.npy',s)
e.close()
