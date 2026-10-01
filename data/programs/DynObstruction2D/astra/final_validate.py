from env_client import make_env
from approach import GeneratedApproach
import sys,time,json
start=int(sys.argv[1]) if len(sys.argv)>1 else 0
end=int(sys.argv[2]) if len(sys.argv)>2 else 100
results=[]
for seed in range(start,end):
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);ts=time.time()
 for k in range(e.max_steps):
  s,r,t,tr,i=e.step(p.get_action(s))
  if t or tr:break
 out={'seed':seed,'success':t,'steps':k+1,'seconds':round(time.time()-ts,2)}
 if not t:
  out['phase']=p.phase;out['state']={n:{f:round(s.get(s.get_object_from_name(n),f),4) for f in ['x','y','theta']} for n in s.get_object_names()}
 print(json.dumps(out),flush=True);results.append(out);e.close()
print('TOTAL',sum(x['success'] for x in results),len(results),flush=True)
