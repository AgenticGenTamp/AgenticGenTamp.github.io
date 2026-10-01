from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor
import time,json

def run(seed):
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{})
 p.reset(s,info);start=time.time()
 def snap(s):
  return {n:[round(float(s.get(s.get_object_from_name(n),f)),4) for f in ['x','y','theta']+([] if n=='robot' else ['width','height'])] for n in s.get_object_names()}
 initial=snap(s);phases=[]
 for k in range(e.max_steps):
  phase=getattr(p,'phase','?')
  if not phases or phases[-1][1]!=phase:phases.append((k,phase))
  a=p.get_action(s);s,r,t,tr,i=e.step(a)
  if t or tr:break
 out={'seed':seed,'steps':k+1,'success':bool(t),'seconds':round(time.time()-start,2)}
 if not t:out.update(initial=initial,final=snap(s),phases=phases)
 e.close();print(json.dumps(out),flush=True);return out
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,range(100,150)))
 with open('scene_validation_results.json','w') as f:json.dump(results,f,indent=2)
 print('SUMMARY',sum(x['success'] for x in results),len(results),flush=True)
