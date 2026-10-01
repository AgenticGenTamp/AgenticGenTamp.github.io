import numpy as np
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env
seeds=[143159,132947,39428,123628,143826,89484,23066,212649,140180,159926,124146,21104]
def run(seed):
 env=make_env();s,_=env.reset(seed=seed)
 v=s[19:21]-np.array([1.15,-2.1]);rad=np.linalg.norm(v);theta=-np.arctan2(v[0],v[1]);goal=s[19:21]+.049999*v/rad
 for k in range(8):
  a=np.zeros(11,dtype=np.float32);a[2]=np.clip(theta-s[2],-.4,.4)
  s,r,t,tr,i=env.step(a)
  if abs(s[2]-theta)<1e-5:break
 for k in range(8):
  a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(goal-s[:2],-.4,.4)
  n,r,t,tr,i=env.step(a)
  if t or tr or np.max(np.abs(n-s))<1e-6:s=n;break
  s=n
 print('ANALYTIC',seed,'radius',rad,'theta',theta,'goal',goal.tolist(),'end',s[:2].tolist(),'success',t,flush=True)
 env.close()
with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,seeds))
