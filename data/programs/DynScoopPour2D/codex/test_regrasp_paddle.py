import math,sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

e=make_env();s,info=e.reset(seed=int(sys.argv[1]) if len(sys.argv)>1 else 0);p=GeneratedApproach(e.action_space,e.observation_space,{}) ; p.reset(s,info)
names=[n for n in s.get_object_names() if n.startswith('small_')]
g=lambda n,f:float(s.get(s.get_object_from_name(n),f))
def step(tx,ty,tt,gap,n):
 global s
 for _ in range(n):
  err=(tt-g('robot','theta')+math.pi)%(2*math.pi)-math.pi
  a=np.array([np.clip(tx-g('robot','x'),-.03,.03),np.clip(ty-g('robot','y'),-.03,.03),np.clip(err,-.098,.098),0,gap])
  s,*_=e.step(a)
t=0
while p.stage<7 and t<500:s,*_=e.step(p.get_action(s));t+=1
for k in range(6):
 step(1.40,.78,0,0,18)
 step(1.40,.78,-math.pi/2,0,18)
 step(1.40,.78,-math.pi/2,.015,15)
 step(g('hook','x'),.75,-math.pi/2,.015,20)
 step(g('hook','x'),.75,-math.pi/2,-.015,15)
 step(.55,.75,-math.pi/2,0,35)
 step(1.40,.75,-math.pi/2,0,35)
 print(k,'right',sum(g(n,'x')>2 for n in names),'held',g('hook','held'),'hook',g('hook','x'),g('hook','y'),flush=True)
e.close()
