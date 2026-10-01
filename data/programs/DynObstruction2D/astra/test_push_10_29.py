from env_client import make_env
from approach import GeneratedApproach
import time
for seed in range(10,30):
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time();trail=[]
 def pos(s,name):
  ob=s.get_object_from_name(name);return [round(s.get(ob,f),3) for f in ['x','y','theta']]
 initial={name:pos(s,name) for name in s.get_object_names()}
 for k in range(e.max_steps):
  s,r,t,tr,i=e.step(p.get_action(s))
  if k%25==0:trail.append((k,p.phase,pos(s,'target_block'),pos(s,'robot')))
  if t or tr:break
 if not t:print('FAIL',seed,k+1,'initial',initial,'trail',trail,flush=True)
 else:print('OK',seed,k+1,round(time.time()-start,2),flush=True)
 e.close()
