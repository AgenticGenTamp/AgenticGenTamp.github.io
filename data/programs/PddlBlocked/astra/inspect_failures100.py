from env_client import make_env
from approach import GeneratedApproach
import numpy as np,sys
for seed in map(int,sys.argv[1:]):
 e=make_env();s,info=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,info)
 print('SEED',seed,'g',a.g,'b',a.b,'d',a.d,'yaw',a.yaw,'candidates',len(a.candidates),flush=True)
 for n in range(400):
  act=a.get_action(s);s,_,t,tr,_=e.step(act)
  if t or tr:break
  if a.stuck==2:
   p=np.array([s.get(a.r,f) for f in a.fs]); print('STUCK',n,'stage',a.stage,'attempt',a.attempt,'held',[name for name in ['blocker','green0'] if s.get(s.get_object_from_name(name),'grasp_active')>.5],'p',p.round(4).tolist(),'queue0',a.queue[0][0].round(4).tolist(),'d',act.round(4).tolist(),flush=True)
 print('END',n+1,t,'stage',a.stage,'attempt',a.attempt,flush=True);e.close()
