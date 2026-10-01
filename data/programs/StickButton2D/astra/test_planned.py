from env_client import make_env
from approach_planned import GeneratedApproach
import sys,time,json
seeds=list(map(int,sys.argv[1:])) or list(range(10))
for seed in seeds:
 env=make_env();s,i=env.reset(seed=seed)
 p=GeneratedApproach(env.action_space,env.observation_space,{})
 p.reset(s,i);start=time.monotonic();last=None
 for k in range(env.max_steps):
  a=p.get_action(s);s,r,term,trunc,i=env.step(a)
  if '--debug' in []:pass
  if term or trunc:break
 remaining=[n for n in s.get_object_names() if n.startswith('button') and s.get(s.get_object_from_name(n),'color_g')<.5]
 print(json.dumps({'seed':seed,'steps':k+1,'success':term,'remaining':remaining,'phase':p.phase,'goal':None if p.goal is None else p.goal.tolist(),'robot':[s.get(s.get_object_from_name('robot'),f) for f in ['x','y','theta','arm_joint']],'time':round(time.monotonic()-start,3)}),flush=True)
 env.close()
