from env_client import make_env
from approach import GeneratedApproach
import sys,time,numpy as np
for seed in [int(x) for x in sys.argv[1:]] or [0,1,2]:
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':20});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time();last=0
 for i in range(e.max_steps):
  a=p.get_action(s);s,r,d,tr,_=e.step(a)
  n=sum(p.done(s,o) for o in p.cubes)
  if n!=last:print('PROGRESS',seed,i,info['object_count'],n,flush=True);last=n
  if d or tr:break
 print('RESULT',seed,info,'steps',i+1,'success',d,'done',last,'time',round(time.time()-start,2),'attempts',p.attempts,flush=True)
 if not d:
  print('UNSORTED',[(o.name,p.xyz(s,o).round(3).tolist(),p.targets[o.name].round(3).tolist()) for o in p.cubes if not p.done(s,o)],flush=True)
 e.close()
