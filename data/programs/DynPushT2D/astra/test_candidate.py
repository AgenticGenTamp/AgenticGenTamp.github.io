from env_client import make_env
from candidate import GeneratedApproach
from approach import wrap
import numpy as np
import sys,time,json
seeds=list(map(int,sys.argv[1:])) or list(range(10))
for seed in seeds:
 e=make_env();s,i=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{}) ;a.reset(s,i);t=time.time();hist=[s.tolist()]
 for k in range(e.max_steps):
  u=a.get_action(s);s,r,term,trunc,i=e.step(u);hist.append(s.tolist())
  if term or trunc:break
 print(seed,k+1,term,'err',np.round(s[29:31]-s[:2],3),round(wrap(s[31]-s[2]),3),'time',round(time.time()-t,2),flush=True)
 with open('trace_%d.json'%seed,'w') as f:json.dump(hist,f)
 e.close()
