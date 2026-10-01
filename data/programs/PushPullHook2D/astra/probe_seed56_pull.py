from env_client import make_env
from approach import GeneratedApproach
import numpy as np
e=make_env();s,i=e.reset(seed=56);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
for t in range(130):
 a=p.get_action(s);s,*_=e.step(a)
 if p.phase>0:print('normal attached',t,flush=True);break
print('before',s[:12].tolist(),flush=True)
for t in range(8):
 old=s.copy();u=np.array([np.cos(s[2]),np.sin(s[2])]);s,*_=e.step(np.array([*(-.005*u),0,0,1],np.float32));print(t,'robotdelta',(s[:2]-old[:2]).tolist(),'hookdelta',(s[9:12]-old[9:12]).tolist(),flush=True)
e.close()
