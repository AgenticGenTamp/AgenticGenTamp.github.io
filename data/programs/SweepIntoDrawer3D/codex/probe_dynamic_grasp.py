import numpy as np
from env_client import make_env
for bx in [1.30,1.36]:
 for by in [-.34,-.38,-.42]:
  e=make_env();o,_=e.reset(seed=0);wp=o[147:150].copy()
  for k in range(12):
   a=np.zeros(11,np.float32);a[:2]=np.clip((np.array([bx,by])-o[125:127])*.5,-.04,.04);a[4]=.025;a[10]=0;o,*_=e.step(a)
  for k in range(4):a=np.zeros(11,np.float32);a[10]=1;o,*_=e.step(a)
  before=o[147:150].copy()
  for k in range(8):a=np.zeros(11,np.float32);a[0]=.06;a[10]=1;o,*_=e.step(a)
  print(bx,by,'contact',np.linalg.norm(before-wp),'follow',np.linalg.norm(o[147:150]-before),o[147:150]);e.close()
