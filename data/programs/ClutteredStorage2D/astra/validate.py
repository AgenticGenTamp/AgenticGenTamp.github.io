import sys,time,json
from env_client import make_env
from approach import GeneratedApproach
startseed=int(sys.argv[1]) if len(sys.argv)>1 else 10
endseed=int(sys.argv[2]) if len(sys.argv)>2 else 40
for seed in range(startseed,endseed):
 e=make_env();s,info=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info);st=time.time();done=False
 for i in range(e.max_steps):
  ac=a.get_action(s);s,r,done,trunc,_=e.step(ac)
  if done or trunc or time.time()-st>65:break
 out={'seed':seed,'count':info['object_count'],'steps':i+1,'success':done,'time':round(time.time()-st,2),'phase':a.phase,'target':a.target,'finished':len(a.finished)}
 if not done:out.update(q=a.q.tolist(),blocks=a.blocks,failures=a.failures,held=a.held)
 print(json.dumps(out),flush=True);e.close()
