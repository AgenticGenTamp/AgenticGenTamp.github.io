from env_client import make_env
from approach import GeneratedApproach
from concurrent.futures import ThreadPoolExecutor
import sys,time,json

def run(args):
 seed,count=args;e=make_env()
 try:
  s,i=e.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i);start=time.time();events=[];last=''
  for k in range(e.max_steps):
   a=p.get_action(s);s,r,t,tr,i=e.step(a)
   if p.phase!=last:events.append([k,p.phase,p.cube.name,p.xyz(s,p.cube).round(3).tolist()]);last=p.phase
   if t or tr:break
   if p.phase=='select' and all(c.name in p.done for c in p.cubes):break
  result={'seed':seed,'count':count,'steps':k+1,'success':t,'seconds':round(time.time()-start,2),'cubes':[p.xyz(s,c).round(4).tolist() for c in p.cubes],'bin':p.xyz(s,p.bins[0]).round(4).tolist(),'events':events}
  return result
 finally:e.close()
if __name__=='__main__':
 start=int(sys.argv[1]) if len(sys.argv)>1 else 0;n=int(sys.argv[2]) if len(sys.argv)>2 else 10
 with ThreadPoolExecutor(max_workers=4) as pool:
  for out in pool.map(run,[(s,c) for s in range(start,start+n) for c in [1,2]]):print(json.dumps(out),flush=True)
