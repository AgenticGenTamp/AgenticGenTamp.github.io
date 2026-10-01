import numpy as np
from env_client import make_env
e=make_env();o,_=e.reset(seed=0);w=o[147:150].copy();q=o[128:135].copy();q[1]=-.28;q[6]=0
for k in range(130):
 a=np.zeros(11,np.float32);a[:2]=np.clip((np.array([1.30,-.50])-o[125:127])*.4,-.03,.03);a[3:10]=np.clip((q-o[128:135])*.5,-.08,.08);a[10]=0;o,*_=e.step(a)
for bx in np.linspace(1.12,1.34,6):
 for by in np.linspace(-.60,-.40,6):
  for k in range(8):
   a=np.zeros(11,np.float32);a[:2]=np.clip((np.array([bx,by])-o[125:127])*.8,-.06,.06);a[10]=0;o,*_=e.step(a)
  before=o[147:150].copy()
  for k in range(4):a=np.zeros(11,np.float32);a[10]=1;o,*_=e.step(a)
  d=np.linalg.norm(o[147:150]-before)
  print(round(bx,2),round(by,2),round(float(d),4))
  if d>.002:print('HIT',o[125:127],o[147:150]);raise SystemExit
  for k in range(2):a=np.zeros(11,np.float32);a[10]=0;o,*_=e.step(a)
e.close()
