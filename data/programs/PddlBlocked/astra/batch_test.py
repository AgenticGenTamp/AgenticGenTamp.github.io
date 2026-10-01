from env_client import make_env
from approach import GeneratedApproach
import time,sys,numpy as np,json
startseed=int(sys.argv[1]) if len(sys.argv)>1 else 0
count=int(sys.argv[2]) if len(sys.argv)>2 else 30
res=[]
for seed in range(startseed,startseed+count):
 e=make_env();s,i=e.reset(seed=seed);start=time.time();a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,i)
 for n in range(400):
  act=a.get_action(s);s,_,t,tr,_=e.step(act)
  if t or tr:break
 rec={'seed':seed,'ok':bool(t),'steps':n+1,'sec':round(time.time()-start,3),'stage':a.stage,'attempt':a.attempt,'candidates':len(a.candidates),'spares':i['object_count']};res.append(rec)
 print(json.dumps(rec),flush=True);e.close()
print('SUMMARY',sum(r['ok'] for r in res),'/',len(res),'mean_steps',round(np.mean([r['steps'] for r in res if r['ok']]),1),flush=True)
