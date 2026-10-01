import sys, numpy as np
from env_client import make_env
from obstacle_alt import GeneratedApproach
for seed in [int(x) for x in sys.argv[1:]]:
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{ });p.reset(s,info)
 last=None
 for t in range(e.max_steps):
  s,re,term,tr,info=e.step(np.asarray(p.get_action(s),dtype=e.action_space.dtype))
  if p.phase!=last:
   obs=[]
   for o in s.get_objects(e.observation_space.get_type('dyn_rectangle')):
    if getattr(o,'name','')!='target_block': obs.append((o.name,round(float(s.get(o,'x')),2),round(float(s.get(o,'y')),2)))
   print(seed,t,p.phase,obs,'b',round(float(s.get(b,'x')),2) if 'b' in locals() else '',flush=True);last=p.phase
  if term or tr:break
 b=s.get_objects(e.observation_space.get_type('target_block'))[0]
 print('RESULT',seed,bool(term),t+1,'block',round(float(s.get(b,'x')),2),round(float(s.get(b,'y')),2),flush=True);e.close()
