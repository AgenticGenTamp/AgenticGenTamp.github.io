from env_client import make_env
from approach import GeneratedApproach
import concurrent.futures,json,sys

def run(args):
 seed,count=args;E=make_env();s,i=E.reset(seed=seed,options={'object_count':count});p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,i)
 for t in range(1000):
  s,r,term,trunc,i=E.step(p.get_action(s))
  if term or trunc:break
 out={'seed':seed,'count':count,'ok':term,'steps':t+1,'phase':p.phase,'held':s.get(p.hook,'held')}
 print(json.dumps(out),flush=True);E.close();return out
if __name__=='__main__':
 start=int(sys.argv[1]) if len(sys.argv)>1 else 0
 end=int(sys.argv[2]) if len(sys.argv)>2 else 10
 counts=tuple(map(int,sys.argv[3:])) or (1,3)
 cases=[(seed,count) for seed in range(start,end) for count in counts]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:res=list(pool.map(run,cases))
 print('TOTAL',sum(r['ok'] for r in res),'/',len(res))
