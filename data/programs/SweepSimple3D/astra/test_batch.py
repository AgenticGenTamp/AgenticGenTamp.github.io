from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor
import time

def run(task):
 seed,n=task;e=make_env();s,i=e.reset(seed=seed,options={'object_count':n});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);ts=time.time()
 for t in range(1000):
  s,r,d,tr,i=e.step(p.get_action(s))
  if d or tr:break
 print('RESULT',seed,n,t+1,d,tr,r,'SEC',round(time.time()-ts,2),'CUBES',[(o.name,round(s.get(o,'x'),2),round(s.get(o,'y'),2)) for o in p.cubes],flush=True);e.close()
with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(run,[(0,1),(2,1),(3,1),(0,5),(2,5),(3,10),(1,50)]))
