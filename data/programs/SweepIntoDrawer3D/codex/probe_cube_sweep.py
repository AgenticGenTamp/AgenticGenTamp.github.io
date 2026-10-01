import numpy as np
from env_client import make_env
e=make_env(); o,_=e.reset(seed=0); init=o.copy(); home=o[128:135].copy()
def go(base=None,q=None,g=0,n=30):
 global o
 for _ in range(n):
  a=np.zeros(11,np.float32); a[10]=g
  if base is not None:a[:3]=np.clip((np.array(base)-o[125:128])*.8,-.1,.1)
  if q is not None:a[3:10]=np.clip((np.array(q)-o[128:135])*.6,-.1,.1)
  o,r,t,tr,info=e.step(a)
go([init[125],-1.35,init[127]],n=20); go([.9,-1.35,np.pi/2],n=25); go([.9,-.92,np.pi/2],n=12)
q=home.copy();q[1]=-1.15;q[3]=-3;go(q=q,g=1,n=15); print('opened',o[103:109],o[125:128])
orig=o[:80].reshape(5,16)[:,:3].copy()
for q2 in [-.8,-.5,-.2,.1]:
 for q4 in [-2.58,-2.0,-1.5]:
  q=home.copy();q[1]=q2;q[3]=q4;q[0]=-1.0;go(q=q,g=0,n=18)
  q[0]=1.0;go(q=q,g=0,n=25)
  xyz=o[:80].reshape(5,16)[:,:3]; d=np.max(np.linalg.norm(xyz-orig,axis=1))
  print('try',q2,q4,'d',round(d,3),'xyz',np.round(xyz,2))
  if d>.03: raise SystemExit
e.close()
