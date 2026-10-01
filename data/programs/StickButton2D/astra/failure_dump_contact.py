from env_client import make_env
from approach import GeneratedApproach
import sys,json
for seed in map(int,sys.argv[1:]):
 env=make_env();s,i=env.reset(seed=seed);p=GeneratedApproach(env.action_space,env.observation_space,{});p.reset(s,i)
 last=None
 for k in range(env.max_steps):
  a=p.get_action(s);s,r,t,u,i=env.step(a)
  if t or u:break
 out={'seed':seed,'steps':k+1,'success':t,'phase':p.phase,'goal':None if p.goal is None else p.goal.tolist(),'target':p.target}
 for n in s.get_object_names():
  o=s.get_object_from_name(n)
  out[n]={f:s.get(o,f) for f in (['x','y','theta','arm_joint'] if n=='robot' else ['x','y','theta','width','height'] if n=='stick' else ['x','y','color_g'])}
 print(json.dumps(out),flush=True);env.close()
