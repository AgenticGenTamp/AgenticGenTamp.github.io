import numpy as np
from env_client import make_env
for tx in [.58,.64,.70,.76,.82]:
 e=make_env();o,_=e.reset(seed=0)
 def go(t,n):
  global o
  for _ in range(n):
   a=np.zeros(11,np.float32);a[:3]=np.clip((np.array(t)-o[125:128])*.8,-.1,.1);a[10]=1;o,*_=e.step(a)
 go([o[125],-1.35,o[127]],20);go([.78,-1.35,1.74],28);go([.78,-.92,1.74],15)
 pre=o[103:109].copy();go([tx,-.92,1.74],80)
 print(tx,'pre',np.round(pre,2),'post',np.round(o[103:109],2),'pose',np.round(o[125:128],2));e.close()
