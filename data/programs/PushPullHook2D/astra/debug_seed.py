from env_client import make_env
from approach import GeneratedApproach
import sys,numpy as np
seed=int(sys.argv[1]);e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
print('initial',s.tolist())
oldphase=-1
for t in range(400):
 a=p.get_action(s)
 if oldphase!=p.phase or t%25==0:
  print(t,'phase',p.phase,'r',s[:5].round(4),'h',s[9:12].round(4),'b',s[20:22].round(4),'a',a.round(4), 'corner',getattr(p,'corner',None),flush=True)
 oldphase=p.phase
 s,r,d,tr,i=e.step(a)
 if d or tr: print('done',t,d);break
e.close()
