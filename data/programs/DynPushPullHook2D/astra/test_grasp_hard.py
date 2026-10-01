from env_client import make_env
from grasp import HookGrasp
for seed in [3,4,6,11,12,21,23,26,27]:
 e=make_env();s,i=e.reset(seed=seed);p=HookGrasp(e.action_space,e.observation_space);p.reset(s)
 h=s.get_objects(e.observation_space.get_type('hook'))[0]
 for k in range(350):
  s,r,d,tr,i=e.step(p.get_action(s))
  if s.get(h,'held') or d or tr:break
 print(seed,k+1,'held',s.get(h,'held'),'phase',p.phase,'attempts',p.attempt,flush=True)
 e.close()
