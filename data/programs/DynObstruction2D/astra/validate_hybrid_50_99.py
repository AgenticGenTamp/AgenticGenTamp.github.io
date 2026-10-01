from env_client import make_env
from approach import GeneratedApproach
import time
for seed in range(50,100):
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time();trail=[]
 def data(s,name):
  ob=s.get_object_from_name(name); fs=['x','y','theta'] if name=='robot' else ['x','y','theta','width','height'];return [round(s.get(ob,f),4) for f in fs]
 initial={name:data(s,name) for name in s.get_object_names()}
 for k in range(e.max_steps):
  s,r,t,tr,i=e.step(p.get_action(s))
  if k%50==0:trail.append((k,p.phase,data(s,'target_block'),data(s,'robot')))
  if t or tr:break
 if not t:print('FAIL',seed,k+1,'initial',initial,'final',{name:data(s,name) for name in s.get_object_names()},'trail',trail,flush=True)
 else:print('OK',seed,k+1,round(time.time()-start,2),flush=True)
 e.close()
