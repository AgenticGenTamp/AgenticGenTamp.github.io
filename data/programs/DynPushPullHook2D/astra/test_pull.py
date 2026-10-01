from env_client import make_env
from grasp import HookGrasp
from pull import HookPull
import sys,time
for seed in map(int,sys.argv[1:] or range(10)):
 e=make_env();s,i=e.reset(seed=seed);g=HookGrasp(e.action_space,e.observation_space);g.reset(s);p=HookPull(e.action_space,e.observation_space);p.reset(s)
 h=s.get_objects(e.observation_space.get_type('hook'))[0];t=s.get_objects(e.observation_space.get_type('target_block'))[0]
 begin=time.time();held=False
 for k in range(650):
  newheld=bool(s.get(h,'held'))
  if newheld and not held:p.reset(s)
  held=newheld
  s,r,d,tr,i=e.step(p.get_action(s) if held else g.get_action(s))
  if k%100==99:print('progress',seed,k+1,'held',held,'stage',p.stage,'target',[round(s.get(t,f),3) for f in ('x','y')],flush=True)
  if d or tr:break
 print(seed,k+1,d,tr,'held',held,'stage',p.stage,'target',[round(s.get(t,f),3) for f in ('x','y')],'secs',round(time.time()-begin,2),flush=True);e.close()
