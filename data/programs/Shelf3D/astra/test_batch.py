from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor
import numpy as np,time,sys

def run(arg):
 seed,count=arg;e=make_env();s,info=e.reset(seed=seed,options={'object_count':count} if count else None);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info);start=time.time();history=[];prev=''
 for i in range(1000):
  a=p.get_action(s)
  if p.phase!=prev:history.append((i,p.index,p.phase));prev=p.phase
  s,r,t,tr,info=e.step(a)
  if t or tr:break
 out={'seed':seed,'count':info['object_count'],'steps':i+1,'success':t,'time':round(time.time()-start,2),'phase':p.phase,'index':p.index,'objects':{o.name:np.round([s.get(o,f) for f in ['x','y','z']],3).tolist() for o in p.objects},'history':history}
 print(out,flush=True);e.close();return out
if __name__=='__main__':
 jobs=[(s,None) for s in range(8)] if len(sys.argv)<2 else [(42,int(c)) for c in sys.argv[1:]]
 with ThreadPoolExecutor(4) as ex:list(ex.map(run,jobs))
