import numpy as np
from env_client import make_env
e=make_env();o,_=e.reset(seed=0);home=o[128:135].copy();w=o[147:150].copy()
def base_goal(t,n):
 global o
 for _ in range(n):
  a=np.zeros(11,np.float32);a[:2]=np.clip((np.array(t)-o[125:127])*2,-.1,.1);a[10]=0;o,*_=e.step(a)
base_goal([o[125],-.75],60);base_goal([1.24,-.75],60);base_goal([1.24,-.38],60)
rng=np.random.default_rng(4)
for i in range(30):
 q=home+rng.uniform([-1,-.8,-.5,-.8,-.7,-.7,-1],[1,.8,.5,.8,.7,.7,1])
 for _ in range(35):
  a=np.zeros(11,np.float32);a[3:10]=np.clip((q-o[128:135])*.5,-.1,.1);a[10]=0;o,*_=e.step(a)
 d=np.linalg.norm(o[147:150]-w)
 if d>.004:
  print('CONTACT',i,d,np.round(o[128:135],3),o[147:150])
  for _ in range(6):a=np.zeros(11,np.float32);a[10]=1;o,*_=e.step(a)
  before=o[147:150].copy()
  for _ in range(10):a=np.zeros(11,np.float32);a[0]=.04;a[4]=.03;a[10]=1;o,*_=e.step(a)
  print('FOLLOW',np.linalg.norm(o[147:150]-before),o[147:150]);break
else:print('none')
e.close()
