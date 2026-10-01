from env_client import make_env
from approach import GeneratedApproach
import sys,time,concurrent.futures,json

def run(seed):
 E=make_env();s,info=E.reset(seed=seed);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info);start=time.monotonic();ph=[]
 for t in range(1000):
  old=p.phase;a=p.get_action(s);s,r,term,trunc,i=E.step(a)
  if p.phase!=old:ph.append((t,p.phase))
  if term or trunc:break
 small=[o for typ in ('small_circle','small_square') for o in s.get_objects(E.observation_space.get_type(typ))]
 out={'seed':seed,'n':len(small),'steps':t+1,'ok':term,'time':round(time.monotonic()-start,2),'phase':p.phase,'right':sum(s.get(o,'x')>1.8 for o in small),'xy':[(round(s.get(o,'x'),2),round(s.get(o,'y'),2)) for o in small] if not term else []}
 print(json.dumps(out),flush=True);E.close();return out
if __name__=='__main__':
 seeds=list(map(int,sys.argv[1:])) or list(range(10))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,seeds))
 print('TOTAL',sum(r['ok'] for r in results),'/',len(results),flush=True)
