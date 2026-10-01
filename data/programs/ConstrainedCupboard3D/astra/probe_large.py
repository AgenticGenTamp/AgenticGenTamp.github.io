from approach import GeneratedApproach
from env_client import make_env
from concurrent.futures import ThreadPoolExecutor
import numpy as np,time

def trial(count):
 e=make_env();s,i=e.reset(seed=0,options={'object_count':count});a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,i);last=None;start=time.time()
 print('INITIAL',count,'actual',len(a.objects),'fixtures',[(f.name,np.round(a.xyz(s,f),3).tolist()) for f in a.fixtures],flush=True)
 for step in range(1000):
  s,r,te,tr,_=e.step(a.get_action(s));status=(a.index,a.stage,len(a.queue))
  if status!=last or te:
   if a.stage==2 or te:print('LARGE',count,step,status,'te',te,[(o.name,np.round(a.xyz(s,o),3).tolist()) for o in a.objects],flush=True)
   last=status
  if te or tr:break
 print('FINAL',count,step+1,te,tr,'elapsed',round(time.time()-start,2),flush=True);e.close()
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(trial,[4,5]))
