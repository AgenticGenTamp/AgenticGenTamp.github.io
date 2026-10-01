import math, sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

seed=int(sys.argv[1]) if len(sys.argv)>1 else 12
e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
names=[n for n in s.get_object_names() if n.startswith('small_')]
g=lambda n,f:float(s.get(s.get_object_from_name(n),f))
def go(x,y,th,n):
 global s
 for _ in range(n):
  er=(th-g('robot','theta')+math.pi)%(2*math.pi)-math.pi
  s,r,d,tr,i=e.step(np.array([np.clip(x-g('robot','x'),-.03,.03),np.clip(y-g('robot','y'),-.03,.03),np.clip(er,-.098,.098),0,0]))
  if d:return True
 return False
t=0
while p.stage<8 and t<500:s,*_=e.step(p.get_action(s));t+=1
for k in range(6):
 go(1.4,g('robot','y'),0,24)
 print(' thrown',k,sum(g(n,'x')>2 for n in names),flush=True)
 go(1.4,1.3,0,20)
 print(' lifted',k,sum(g(n,'x')>2 for n in names),flush=True)
 go(.55,1.3,-math.pi/2,35)
 print(' reset',k,sum(g(n,'x')>2 for n in names),flush=True)
 floor_y=np.clip(g('robot','y')+.03-g('hook','y'),.72,1.3)
 go(.55,floor_y,-math.pi/2,25);go(1.4,floor_y,-math.pi/2,35)
 t+=139
 print(k,t,'right',sum(g(n,'x')>2 for n in names),'held',g('hook','held'),'hook',round(g('hook','x'),2),round(g('hook','y'),2),flush=True)
e.close()
