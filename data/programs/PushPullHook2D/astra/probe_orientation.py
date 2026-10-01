from env_client import make_env
from approach import GeneratedApproach
import numpy as np
for mode in ['vacoff','detect']:
 e=make_env();s,i=e.reset(seed=84);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 last=-9
 for t in range(350):
  if mode=='detect' and p.phase==-1 and np.linalg.norm(s[9:12]-p.initial_hook)>.001:p.phase=1
  a=p.get_action(s)
  if mode=='vacoff' and p.phase==-1:a[4]=0
  if p.phase!=last or t%50==0:print(mode,t,'phase',p.phase,'r',s[:5].round(3),'hook',s[9:12].round(3),'a',a.round(3),flush=True)
  last=p.phase;s,r,d,tr,i=e.step(a)
  if d or tr:print(mode,'done',d,t,flush=True);break
 e.close()
