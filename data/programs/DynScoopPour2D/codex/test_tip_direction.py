import math,sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
tt=float(sys.argv[1])
e=make_env();s,info=e.reset(seed=0);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
names=[n for n in s.get_object_names() if n.startswith('small_')]
g=lambda n,f:float(s.get(s.get_object_from_name(n),f))
while p.stage<7:s,*_=e.step(p.get_action(s))
for k in range(60):
 a=np.array([0,np.clip(2.4-g('robot','y'),-.03,.03),0,0,0]);s,*_=e.step(a)
for k in range(30):
 err=(tt-g('robot','theta')+math.pi)%(2*math.pi)-math.pi
 s,r,d,tr,i=e.step(np.array([0,0,np.clip(err,-.098,.098),0,0]))
 print(k,round(g('robot','theta'),2),round(g('hook','x'),2),sum(g(n,'x')>2 for n in names),sum(g(n,'y')>1.5 for n in names),d,flush=True)
 if d:break
for k in range(20):
 s,r,d,tr,i=e.step(np.array([0,0,0,.08,0]))
 print('extend',k,round(g('robot','arm_joint'),2),round(g('hook','x'),2),sum(g(n,'x')>2 for n in names),sum(g(n,'y')>1.5 for n in names),d,flush=True)
 if d:break
e.close()
