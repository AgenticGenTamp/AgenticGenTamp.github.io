from env_client import make_env
from approach import GeneratedApproach
import numpy as np,sys,time,json
seeds=list(map(int,sys.argv[1:])) or list(range(6))
for seed in seeds:
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);st=time.time();lastphase=None
 for k in range(e.max_steps):
  s,r,t,tr,i=e.step(p.get_action(s))
  key=(p.which,p.phase,p.retries)
  if key!=lastphase:
   print('PHASE',seed,k,key,'xyz',np.round(s[p.obj:p.obj+3],4),flush=True);lastphase=key
  if t or tr:break
 print('RESULT',seed,k+1,t,tr,r,i,'time',round(time.time()-st,2),'objects',np.round(s[[0,1,2,54,55,56,70,71,72]],4),'beam',np.round(s[38:45],5),flush=True)
 json.dump(s.tolist(),open('end_seed'+str(seed)+'.json','w'));e.close()
