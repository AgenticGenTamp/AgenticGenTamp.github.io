import numpy as np
from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor

def trial(mode):
 e=make_env()
 try:
  s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);c=p.cube
  for t in range(140):s,*_=e.step(p.get_action(s))
  vals=[]
  for t in range(12):
   a=np.zeros(18,np.float32)
   if mode=='before' and t==0:a[10]=1;a[12]=-6
   if mode=='during' and t==0:a[12]=-6
   s,r,d,tr,info=e.step(a)
   vals.append([t]+[round(s.get(c,f),4) for f in ['x','y','z','vx','vy','vz']])
  return mode,vals
 finally:e.close()
with ThreadPoolExecutor(max_workers=3) as pool:
 for mode,vals in pool.map(trial,['drop','before','during']):
  print(mode,flush=True)
  for row in vals:print(row,flush=True)
