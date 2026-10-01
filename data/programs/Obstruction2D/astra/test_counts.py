from env_client import make_env
from approach import GeneratedApproach
import sys,time
count=int(sys.argv[1]);seeds=[int(x) for x in sys.argv[2:]] or list(range(30))
for seed in seeds:
 e=make_env()
 try:s,info=e.reset(seed=seed,options={'object_count':count})
 except Exception as ex:
  print('RESET_ERROR',seed,count,str(ex),flush=True);e.close();continue
 p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.monotonic()
 for k in range(e.max_steps):
  a=p.get_action(s);s,r,te,tr,i=e.step(a)
  if te or tr:break
 print('RESULT',seed,'count',count,'steps',k+1,'success',te,'seconds',round(time.monotonic()-start,2),'phase',p.phase,flush=True)
 if not te:
  print('POLICY',str({k:v for k,v in p.__dict__.items() if k not in ['types','action_space']}),flush=True)
  for o in s.get_objects(e.observation_space.get_type('rectangle')):print(o.name,p.rect(s,o),flush=True)
 e.close()
