import numpy as np
from env_client import make_env
e=make_env();o,_=e.reset(seed=0);home=o[128:135].copy()
def go(base=None,q=None,g=1,n=20):
 global o
 for _ in range(n):
  a=np.zeros(11,np.float32);a[10]=g
  if base is not None:a[:3]=np.clip((np.array(base)-o[125:128])*.8,-.1,.1)
  if q is not None:a[3:10]=np.clip((q-o[128:135])*.7,-.1,.1)
  o,*_=e.step(a)
go([o[125],-1.35,o[127]],n=20);go([.9,-1.35,np.pi/2],n=25);go([.9,-.92,np.pi/2],n=12)
q=home.copy();q[1]=-1.15;q[3]=-3;go(q=q,n=15);print('open',o[103:109])
wp=o[147:150].copy(); cubes=o[:80].reshape(5,16)[:,:3].copy(); prev=o[128:135].copy();rng=np.random.default_rng(9)
for i in range(65):
 q=home.copy();q += rng.uniform([-1.5,-1.0,-.7,-.1,-1.2,-1.2,-1.5],[1.5,.8,.7,2.5,1.2,1.2,1.5]);q[3]=np.clip(q[3],-2.58,-.1)
 go(q=q,n=12)
 dw=np.linalg.norm(o[147:150]-wp); dc=np.max(np.linalg.norm(o[:80].reshape(5,16)[:,:3]-cubes,axis=1))
 if dw>.006 or dc>.006:
  print('HIT',i,'prev',np.round(prev,2),'target',np.round(q,2),'actual',np.round(o[128:135],2),'dw',dw,'dc',dc,'w',o[147:150],'c',o[:80].reshape(5,16)[:,:3]);break
 prev=o[128:135].copy()
else:print('none')
e.close()
