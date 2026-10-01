import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True)
e=make_env()
for yo in [0,.15,-.15]:
 for qo in [-.6,-.3,0,.3,.6,.9,1.2]:
  s,_=e.reset(seed=0);q=s[128:135].copy();q[3]+=qo;w=s[147:150].copy();c=s[:80].reshape(5,16)[:,:3].copy();first=False;maxmove=0
  def step(x,n):
   global s,first,maxmove
   for k in range(n):
    a=np.zeros(11,np.float32);a[0]=np.clip(1.2*(x-s[125]),-.05,.05);a[1]=np.clip(1.2*(w[1]+yo-s[126]),-.05,.05);a[2]=np.clip(np.pi-s[127],-.1,.1);a[3:10]=np.clip(2*(q-s[128:135]),-.1,.1);a[10]=1
    s,*_=e.step(a)
    dm=float(np.linalg.norm(s[147:150]-w));dc=float(np.max(np.linalg.norm(s[:80].reshape(5,16)[:,:3]-c,axis=1)));maxmove=max(maxmove,dm)
    if (dm>.003 or dc>.01) and not first:
     first=True;print('CONTACT',yo,qo,'target',x,'base',s[125:128],'joint',s[128:135],'wdelta',s[147:150]-w,'cubes',dc,flush=True);np.save('contact_'+str(yo)+'_'+str(qo)+'.npy',s)
  step(1.8,35);step(.65,45);step(1.8,45)
  print('done',yo,qo,'base',s[125:127],'wiper_delta',s[147:150]-w,'maxmove',maxmove,'drawers',s[103:109],flush=True)
e.close()
