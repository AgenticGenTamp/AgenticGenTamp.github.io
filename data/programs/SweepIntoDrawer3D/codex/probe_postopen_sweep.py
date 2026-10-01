import numpy as np
from env_client import make_env
e=make_env();o,_=e.reset(seed=0);home=o[128:135].copy();orig=o[:80].reshape(5,16)[:,:3].copy()
def go(base=None,q=None,g=1,n=30):
 global o
 for _ in range(n):
  a=np.zeros(11,np.float32);a[10]=g
  if base is not None:a[:3]=np.clip((np.array(base)-o[125:128])*.8,-.1,.1)
  if q is not None:a[3:10]=np.clip((q-o[128:135])*.7,-.1,.1)
  o,r,t,tr,info=e.step(a)
# known opening route
go([o[125],-1.35,o[127]],n=20);go([.9,-1.35,np.pi/2],n=25);go([.9,-.92,np.pi/2],n=12)
print('open',o[103:109])
# return around east end, start north of pile, translate wiper/gripper south
go([.9,-1.65,np.pi/2],n=15);go([1.85,-1.65,np.pi],n=25);go([1.85,.18,np.pi],n=30);go([1.24,.18,np.pi],n=12)
print('ready base',o[125:128],'draw',o[103:109])
go([1.24,-.07,np.pi],home,1,n=12)
for qa in [[-.51,.38,2.89,-2.49,.46,-.25,1.99],[.37,-.44,3.38,-2.58,.6,-.49,1.35]]:
 go([1.24,-.07,np.pi],np.array(qa),0,n=35)
 xyz=o[:80].reshape(5,16)[:,:3]
 print('qpath delta',np.round(xyz-orig,3),'w',np.round(o[147:150],3),'draw',np.round(o[103:109],3))
e.close()
