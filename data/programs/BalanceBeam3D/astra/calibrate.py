from env_client import make_env
from approach import GeneratedApproach
import numpy as np,sys,time,json
for mount in map(float,sys.argv[1:] or ['.3','.4','.5']):
 e=make_env();s,i=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.mount=mount;p.reset(s,i);mx=0
 for k in range(450):
  s,r,t,tr,i=e.step(p.get_action(s));mx=max(mx,s[2])
  if k%100==99:print(mount,k,'phase',p.phase,'large',np.round(s[:3],4),'q',np.round(s[19:27],3),flush=True)
 print('RESULT',mount,mx,s[:3],flush=True);e.close()
