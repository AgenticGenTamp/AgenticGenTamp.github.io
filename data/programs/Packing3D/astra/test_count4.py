from env_client import make_env
from approach import GeneratedApproach
import numpy as np,time
for seed in [0,1,2]:
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':4});p=GeneratedApproach(e.action_space,e.observation_space,{});t=time.time();p.reset(s,info);print('LAYOUT',seed,p.layout,'time',time.time()-t,flush=True)
 same=0
 for i in range(250):
  a=p.get_action(s);ns,r,te,tr,inf=e.step(a)
  same=same+1 if np.max(np.abs(p.config(ns)-p.config(s)))<1e-6 else 0;s=ns
  if te or same>6:break
 print('RESULT',seed,i+1,te,p.phase,p.target.name if p.target else '', 'xyz',p.xyz(s,p.target) if p.target else None,flush=True)
 e.close()
