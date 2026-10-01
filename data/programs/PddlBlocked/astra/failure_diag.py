from env_client import make_env
from approach import GeneratedApproach
import numpy as np,sys
seed=int(sys.argv[1]);e=make_env();s,i=e.reset(seed=seed);a=GeneratedApproach(e.action_space,e.observation_space,{});a.reset(s,i)
print('seed',seed,'candidates',len(a.candidates),'g',a.g,'b',a.b,flush=True)
for n in range(180):
 act=a.get_action(s)
 if a.stuck>=2 or n%20==0:
  print(n,'stage',a.stage,'attempt',a.attempt,'stuck',a.stuck,'p',[round(s.get(a.r,f),3) for f in a.fs],'target',a.queue[0][0].round(3).tolist() if a.queue else None,'red',a.pos(s,'blocker').round(3),'green',a.pos(s,'green0').round(3),flush=True)
 s,_,t,tr,_=e.step(act)
 if t:print('SUCCESS',n,flush=True);break
e.close()
