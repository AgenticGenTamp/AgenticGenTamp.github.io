import numpy as np
from env_client import make_env
e=make_env();o,_=e.reset(seed=0);w=o[147:150].copy();home=o[128:135].copy()
base=home.copy();base[6]=0
for k in range(110):
 a=np.zeros(11,np.float32);a[:2]=np.clip((np.array([1.23,-.51])-o[125:127])*.5,-.04,.04);a[3:10]=np.clip((base-o[128:135])*.6,-.08,.08);a[10]=0;o,*_=e.step(a)
for q1 in [-.6,-.4,-.2,0,.2,.4,.6]:
 for q2 in [-.18,-.34,-.50]:
  q=base.copy();q[0]=q1;q[1]=q2
  for k in range(18):a=np.zeros(11,np.float32);a[3:10]=np.clip((q-o[128:135])*.6,-.06,.06);a[10]=0;o,*_=e.step(a)
  before=o[147:150].copy()
  for k in range(4):a=np.zeros(11,np.float32);a[10]=1;o,*_=e.step(a)
  d=np.linalg.norm(o[147:150]-before)
  print(q1,q2,round(float(d),4),np.round(o[147:150],3))
  if d>.002:print('HIT actual',np.round(o[128:135],3));raise SystemExit
e.close()
