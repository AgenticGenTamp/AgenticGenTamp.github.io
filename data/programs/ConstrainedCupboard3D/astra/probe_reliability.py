from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor
import numpy as np,time

def trial(seed):
 e=make_env();s,info=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info);obj=a.objects[0];initial=a.xyz(s,obj);start=time.time()
 for step in range(700):
  s,r,te,tr,_=e.step(a.get_action(s))
  if a.stage==1 and not a.queue:
   p=a.xyz(s,obj);print('GRASP',seed,'count',len(a.objects),'steps',step+1,'height',round(p[2],4),'pos',np.round(p,4).tolist(),'initial',np.round(initial,4).tolist(),'success',bool(p[2]>.20),'elapsed',round(time.time()-start,2),flush=True);break
  if te or tr: print('TERMINATED',seed,step,te,tr,flush=True);break
 else: print('INCOMPLETE',seed,a.stage,len(a.queue),flush=True)
 e.close()
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(trial,range(10)))
