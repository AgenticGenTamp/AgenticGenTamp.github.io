from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor
import numpy as np,time

def trial(seed):
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':1});a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info);start=time.time();obj=a.objects[0];fixt=[(f.name,np.round(a.xyz(s,f),3).tolist()) for f in a.fixtures]
 for step in range(450):
  s,r,te,tr,_=e.step(a.get_action(s))
  if te or tr or (a.stage==3 and not a.queue):break
 print('SINGLE',seed,'steps',step+1,'terminated',te,'truncated',tr,'reward',r,'pos',np.round(a.xyz(s,obj),4).tolist(),'fixtures',fixt,'elapsed',round(time.time()-start,2),flush=True)
 e.close()
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(trial,range(10)))
