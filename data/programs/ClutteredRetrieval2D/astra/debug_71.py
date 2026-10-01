from env_client import make_env
from approach import GeneratedApproach
import numpy as np,json,time
E=make_env();s,i=E.reset(seed=71);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,i)
for k in range(300):
 a=p.get_action(s)
 if k<60 or k%20==0:
  print(json.dumps({'k':k,'phase':p.phase,'chosen':p.chosen,'q':p.q.tolist(),'action':a.tolist(),'stuck':p.stuck,'path_n':len(p.path),'held':None if p.held is None else p.held.tolist(),'rects':p.rects.tolist() if k==50 else None}),flush=True)
 s,r,d,t,inf=E.step(a)
 if d or t:print('DONE',k,d,t);break
E.close()
