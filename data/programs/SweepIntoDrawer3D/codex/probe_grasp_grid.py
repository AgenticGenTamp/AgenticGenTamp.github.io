import numpy as np
from env_client import make_env
for by in [-.28,-.36]:
 for q1 in [-.25,0,.25]:
  for q2 in [-.35,-.50]:
   e=make_env();o,_=e.reset(seed=0);wp=o[147:150].copy();home=o[128:135].copy()
   def go(base=None,q=None,g=1,n=20):
    global o
    for _ in range(n):
     a=np.zeros(11,np.float32);a[10]=g
     if base is not None:a[:2]=np.clip((np.array(base)-o[125:127])*.8,-.1,.1)
     if q is not None:a[3:10]=np.clip((q-o[128:135])*.7,-.1,.1)
     o,*_=e.step(a)
   go([1.34,by],home,1,12);q=home.copy();q[0]=q1;q[1]=q2;go([1.20,by],q,1,12)
   go([1.20,by],q,0,4);before=o[147:150].copy();go([1.50,by],q,0,8)
   d=np.linalg.norm(o[147:150]-before)
   print(by,q1,q2,'base',np.round(o[125:127],2),'w',np.round(o[147:150],3),'d',round(d,3))
   e.close()
