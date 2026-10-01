import numpy as np
from env_client import make_env
for seed in (72,601,1000738):
 for theta in np.arange(-4,4)*np.pi/4:
  for off in ((.04999,0),(0,.04999),(.03534,.03534)):
   env=make_env();s,_=env.reset(seed=seed)
   for k in range(20):
    a=np.zeros(11,dtype=np.float32);a[2]=np.clip(theta-s[2],-.4,.4)
    s,r,t,tr,i=env.step(a)
    if abs(s[2]-theta)<1e-5:break
   goal=s[19:21]+off
   for k in range(12):
    a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(goal-s[:2],-.4,.4)
    n,r,t,tr,i=env.step(a)
    if t or tr or np.max(np.abs(n-s))<1e-6:s=n;break
    s=n
   print('CASE',seed,round(theta,3),off,'pos',s[:3].tolist(),'term',t,flush=True)
   env.close()
