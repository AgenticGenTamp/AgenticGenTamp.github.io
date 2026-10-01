from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor
import time,sys

def trial(seed):
 e=make_env();s,info=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info);start=time.time();te=tr=False
 for t in range(1000):
  s,r,te,tr,_=e.step(a.get_action(s))
  if te or tr:break
 print('RESULT',seed,info,'done',te,'steps',t+1,'secs',round(time.time()-start,2),'pos',[(o.name,a.xyz(s,o).round(3).tolist()) for o in a.objects],flush=True);e.close()
if __name__=='__main__':
 start=int(sys.argv[1]) if len(sys.argv)>1 else 0
 count=int(sys.argv[2]) if len(sys.argv)>2 else 10
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(trial,range(start,start+count)))
