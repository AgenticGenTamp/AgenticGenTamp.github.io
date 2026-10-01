from env_client import make_env
from approach import GeneratedApproach
import numpy as np

e=make_env();s,info=e.reset(seed=5);a=GeneratedApproach(e.action_space,e.observation_space,{})
a.reset(s,info)
for i in range(1000):
 act=a.get_action(s)
 if i>=350:
  p=a.planner(a.target)
  new=a.q+act[:4]
  print('DIAG',i,'phase',a.phase,'q',a.q,'action',act,'held',a.held,'path',a.path,'nextvalid',p.valid(new),'edgevalid',p.edge(a.q,a.path[0]) if a.path else None,flush=True)
  break
 s,r,done,trunc,inf=e.step(act)
 if done or trunc:break
e.close()

from planner import Planner
import math,time

def fine_edge(self,a,b):
 d=self.delta(a,b);n=max(1,int(math.ceil(max(abs(d[0])/.01,abs(d[1])/.01,abs(d[2])/.03,abs(d[3])/.02))))
 return all(self.valid(a+d*(j/n)) for j in range(1,n+1))
Planner.edge=fine_edge
e=make_env();s,info=e.reset(seed=5);a=GeneratedApproach(e.action_space,e.observation_space,{})
a.reset(s,info);start=time.time()
for i in range(1000):
 act=a.get_action(s);s,r,done,trunc,inf=e.step(act)
 if done or trunc:break
print('FINE_RESULT',5,i+1,done,trunc,'elapsed',time.time()-start,'phase',a.phase,'q',a.q,'target',a.target,flush=True)
e.close()
