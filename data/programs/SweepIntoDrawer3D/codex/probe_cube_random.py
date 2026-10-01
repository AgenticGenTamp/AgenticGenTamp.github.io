import numpy as np
from env_client import make_env
e=make_env();o,_=e.reset(seed=0);home=o[128:135].copy();orig=o[:80].reshape(5,16)[:,:3].copy()
def go(base=None,q=None,g=1,n=20):
 global o
 for _ in range(n):
  a=np.zeros(11,np.float32);a[10]=g
  if base is not None:a[:3]=np.clip((np.array(base)-o[125:128])*.8,-.1,.1)
  if q is not None:a[3:10]=np.clip((q-o[128:135])*.7,-.1,.1)
  o,*_=e.step(a)
go([o[125],-1.35,o[127]],n=20);go([.9,-1.35,np.pi/2],n=25);go([.9,-.92,np.pi/2],n=12)
go([.9,-1.65,np.pi/2],n=15);go([1.85,-1.65,np.pi],n=25);go([1.85,-.07,np.pi],n=30);go([1.24,-.07,np.pi],n=12)
print('ready',o[125:128],o[103:109])
rng=np.random.default_rng(22);prev=o[128:135].copy()
lo=np.array([-2,-1.4,2.0,-2.58,-1.5,-1.8,0.]);hi=np.array([2,1.0,4.2,.2,1.5,.5,3.1])
for i in range(62):
 q=rng.uniform(lo,hi);go(q=q,g=0,n=12)
 xyz=o[:80].reshape(5,16)[:,:3];d=np.max(np.linalg.norm(xyz-orig,axis=1))
 if d>.006:
  print('HIT',i,'prev',np.round(prev,2),'tgt',np.round(q,2),'act',np.round(o[128:135],2),'d',d,'xyz',xyz,'draw',o[103:109]);break
 prev=o[128:135].copy()
else:print('none')
e.close()
