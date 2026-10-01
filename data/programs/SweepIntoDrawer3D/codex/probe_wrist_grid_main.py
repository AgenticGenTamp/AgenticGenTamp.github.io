import numpy as np
from env_client import make_env

e=make_env(); o,_=e.reset(seed=0); initial=o[147:150].copy(); home=o[128:135].copy()
def move(q,g,n=20):
 global o
 for _ in range(n):
  a=np.zeros(11,np.float32); a[:2]=np.clip((np.array([1.23,-.51])-o[125:127])*.5,-.04,.04)
  a[3:10]=np.clip((q-o[128:135])*.6,-.08,.08); a[10]=g; o,*_=e.step(a)
base=home.copy();base[1]=-.28;base[6]=0
move(base,0,120)
for q5 in [-.6,-.3,0,.3,.6]:
 for q6 in [-1.25,-.95,-.65,-.35]:
  q=base.copy();q[4]=q5;q[5]=q6
  move(q,0,18);before=o[147:150].copy();move(q,1,5)
  d=np.linalg.norm(o[147:150]-before);print(q5,q6,round(float(d),4),np.round(o[147:150],3))
  if d>.002:print('HIT',np.round(o[128:135],3));raise SystemExit
e.close()
