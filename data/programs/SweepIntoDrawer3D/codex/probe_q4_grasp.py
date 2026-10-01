import numpy as np
from env_client import make_env
e=make_env();o,_=e.reset(seed=0);w=o[147:150].copy();q=o[128:135].copy();q[1]=-.28;q[6]=0
for k in range(120):
 a=np.zeros(11,np.float32);a[:2]=np.clip((np.array([1.23,-.51])-o[125:127])*.4,-.03,.03);a[3:10]=np.clip((q-o[128:135])*.5,-.08,.08);a[10]=0;o,*_=e.step(a)
for q4 in [-2.58,-2.4,-2.2,-2.0,-1.8,-1.6]:
 q[3]=q4
 for k in range(16):a=np.zeros(11,np.float32);a[3:10]=np.clip((q-o[128:135])*.5,-.05,.05);a[10]=0;o,*_=e.step(a)
 before=o[147:150].copy()
 for k in range(5):a=np.zeros(11,np.float32);a[10]=1;o,*_=e.step(a)
 print(q4,'openmove',np.linalg.norm(before-w),'close',np.linalg.norm(o[147:150]-before),np.round(o[147:150],3))
 if np.linalg.norm(o[147:150]-w)>.003:break
e.close()
