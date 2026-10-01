from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor
import time,sys,numpy as np

def run(seed,count):
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time();hist=[];prev=''
 for i in range(1000):
  a=p.get_action(s)
  if p.phase!=prev:hist.append((i,p.index,p.phase));prev=p.phase
  s,r,t,tr,info=e.step(a)
  if t or tr:break
 print('RESULT',seed,count,t,i+1,round(time.time()-start,2),flush=True)
 if not t:print('FAILDETAIL',seed,p.phase,p.index,{o.name:np.round([s.get(o,f) for f in ['x','y','z']],3).tolist() for o in p.objects},hist,flush=True)
 e.close();return t
if __name__=='__main__':
 count=int(sys.argv[1]) if len(sys.argv)>1 else 8
 seeds=range(int(sys.argv[2]) if len(sys.argv)>2 else 12)
 with ThreadPoolExecutor(4) as ex:results=list(ex.map(lambda s:run(s,count),seeds))
 print('TOTAL',sum(results),len(results),flush=True)
