import math
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

e=make_env();s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{}) ; p.reset(s,info)
names=[n for n in s.get_object_names() if n.startswith('small_')]
g=lambda n,f:float(s.get(s.get_object_from_name(n),f))
t=0
while p.stage < 7 and t<500:
 s,*_=e.step(p.get_action(s));t+=1
print('prepared',t,g('robot','x'),g('robot','y'),sum(g(n,'x')>2 for n in names))
for k in range(30):
 target=0.0 if k%2==0 else -math.pi/2
 for j in range(18):
  err=(target-g('robot','theta')+math.pi)%(2*math.pi)-math.pi
  s,r,done,trunc,info=e.step(np.array([-.03 if g('robot','x')>1.30 else 0,0,np.clip(err,-.098,.098),0,0]));t+=1
 print(k,t,round(g('robot','x'),2),round(g('robot','y'),2),round(g('robot','theta'),2),round(g('hook','x'),2),round(g('hook','y'),2),sum(g(n,'x')>2 for n in names),done,flush=True)
 if done:break
e.close()
