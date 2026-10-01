import numpy as np
from concurrent.futures import ThreadPoolExecutor
from env_client import make_env
candidates=[]
for seed in range(500000):
 t=np.random.default_rng(seed).uniform(-2,2,2)
 if -2<t[1]<-1.959 and 1.20<t[0]<1.30:candidates.append((seed,t))
 if len(candidates)>=60:break
candidates.sort(key=lambda v:v[1][0])
chosen=[candidates[i] for i in np.linspace(0,len(candidates)-1,12,dtype=int)]
print('CHOSEN',[(s,t.tolist()) for s,t in chosen],flush=True)
def run(item):
 seed,target=item;results=[]
 for theta in (-np.pi/4,0,-.2,-.4,-.6,-.8,-1.,-1.2):
  env=make_env();s,_=env.reset(seed=seed)
  assert np.max(np.abs(s[19:21]-target))<1e-6,(seed,s[19:21],target)
  for k in range(6):
   a=np.zeros(11,dtype=np.float32);a[2]=np.clip(theta-s[2],-.4,.4)
   s,r,t,tr,i=env.step(a)
   if abs(s[2]-theta)<1e-5:break
  success=None
  for off in ((0,.049999),(.0353546,.0353546),(.049999,0)):
   goal=target+off
   for k in range(8):
    a=np.zeros(11,dtype=np.float32);a[:2]=np.clip(goal-s[:2],-.4,.4)
    n,r,t,tr,i=env.step(a)
    if t or tr or np.max(np.abs(n-s))<1e-6:s=n;break
    s=n
   if t:success=off;break
  env.close();results.append((theta,success))
 print('SEED',seed,'target',target.tolist(),'results',results,flush=True)
 return seed,target.tolist(),results
with ThreadPoolExecutor(max_workers=3) as pool:
 list(pool.map(run,chosen))
