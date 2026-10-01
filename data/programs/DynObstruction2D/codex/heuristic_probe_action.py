from env_client import make_env
import numpy as np
for a in ([0,0,0,0,0], [.05,0,0,0,0], [0,-.05,0,0,0], [0,0,.196,0,0], [.05,-.05,.196,0,0]):
 e=make_env(); s,_=e.reset(seed=0)
 try:
  s,*rest=e.step(np.asarray(a,dtype=np.float32)); print(a,"ok",rest)
 except Exception as x: print(a,"ERR",repr(x))
 e.close()
