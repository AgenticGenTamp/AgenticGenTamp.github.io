from env_client import make_env
from approach import GeneratedApproach
import time,numpy as np,json,sys
from concurrent.futures import ThreadPoolExecutor

def run(seed):
 e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{})
 p.reset(s,info);start=time.time();t=False;fail=None
 for k in range(e.max_steps):
  s,r,t,tr,info=e.step(p.get_action(s))
  if t or tr:break
 out={'seed':seed,'success':bool(t),'steps':k+1,'sec':round(time.time()-start,2),'phase':p.phase,'item':p.item,'positions':np.round(s[[0,1,2,16,17,18,32,33,34]],4).tolist()}
 e.close();print(json.dumps(out),flush=True)
 if not t:json.dump(s.tolist(),open('failed_seed%d.json'%seed,'w'))
 return out
if __name__=='__main__':
 seeds=list(map(int,sys.argv[1:])) or list(range(10))
 with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,seeds))
 print('TOTAL',sum(x['success'] for x in results),'/',len(results),flush=True)
