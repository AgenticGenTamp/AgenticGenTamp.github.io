from env_client import make_env
from approach import GeneratedApproach
for seed in [63,64]:
 e=make_env();s,info=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info)
 for step in range(65):
  act=a.get_action(s)
  if step>57:print(seed,step,[a.xy(s,r).tolist() for r in a.rovers],act.tolist(),a.goal,flush=True)
  s,*_=e.step(act)
 print('obstacles',seed,[(c.tolist(),h.tolist()) for c,h,z in a.block],flush=True)
 e.close()
