import numpy as np
from env_client import make_env
e=make_env(); o,_=e.reset(seed=0); wp=o[147:150].copy()
def goal(base=None,q=None,g=1,n=80):
 global o
 for k in range(n):
  a=np.zeros(11,np.float32); a[10]=g
  if base is not None: a[:2]=np.clip((np.array(base)-o[125:127])*2,-.1,.1)
  if q is not None: a[3:10]=np.clip((np.array(q)-o[128:135])*.5,-.1,.1)
  o,r,t,tr,info=e.step(a)
  if base is not None and np.linalg.norm(o[125:127]-base)<.025: break
goal([o[125],-.75]); goal([1.24,-.75]); goal([1.24,-.38])
q0=np.array([-.51,.38,2.89,-2.49,.46,-.25,1.99]); q=np.array([.37,-.44,3.38,-2.58,.6,-.49,1.35])
goal(q=q0,n=50); goal(q=q,n=35); print('at',o[147:150],o[128:135])
goal(q=q,g=0,n=5); print('closed',o[147:150],o[135])
goal([1.24,-.58],q=q,g=0,n=30); print('movebase',o[125:127],o[147:150], 'delta',o[147:150]-wp)
goal([1.24,-.38],q=q,g=0,n=30); print('return',o[125:127],o[147:150])
e.close()
