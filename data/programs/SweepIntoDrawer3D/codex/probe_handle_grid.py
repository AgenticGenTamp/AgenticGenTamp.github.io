import numpy as np
from env_client import make_env
e=make_env(); o,_=e.reset(seed=0); home=o[128:135].copy()
def step(base=None,q=None,g=1,n=30):
 global o
 for _ in range(n):
  a=np.zeros(11,np.float32); a[10]=g
  if base is not None: a[:3]=np.clip((np.array(base)-o[125:128])*.8,-.1,.1)
  if q is not None: a[3:10]=np.clip((np.array(q)-o[128:135])*.6,-.1,.1)
  o,r,t,tr,info=e.step(a)
# route south then x=.9 and yaw pi/2
step([o[125],-1.35,o[127]],n=20); step([.9,-1.35,np.pi/2],n=25); step([.9,-.92,np.pi/2],n=12)
print('pose',o[125:128],'pre_draw',o[103:109])
best=0
for q2 in np.linspace(-1.15,.15,6):
 for q4 in np.linspace(-3.0,-1.25,6):
  q=home.copy(); q[1]=q2; q[3]=q4
  step(q=q,g=1,n=12); step([.9,-.75,np.pi/2],q,1,n=5); step(q=q,g=0,n=3); step([.9,-1.1,np.pi/2],q,0,n=6)
  d=np.max(np.abs(o[103:109])); best=max(best,d)
  if d>.003: print('HIT',q2,q4,np.round(o[103:109],4),o[125:128],np.round(o[128:135],2)); raise SystemExit
  step(q=q,g=1,n=2)
print('best',best,'draw',o[103:109]); e.close()
