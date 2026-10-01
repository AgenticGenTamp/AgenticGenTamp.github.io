import math,sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
bx=float(sys.argv[1]); speed=float(sys.argv[2]); seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
names=[n for n in s.get_object_names() if n.startswith('small_')]
g=lambda n,f:float(s.get(s.get_object_from_name(n),f))
while p.stage<7:s,*_=e.step(p.get_action(s))
for _ in range(12):s,*_=e.step(np.array([np.clip(bx-g('robot','x'),-.03,.03),0,0,0,0]))
for k in range(100):
 err=(0-g('robot','theta')+math.pi)%(2*math.pi)-math.pi
 s,r,d,tr,i=e.step(np.array([0,0,np.clip(err,-speed,speed),0,0]))
 if abs(err)<.01 or d:break
for _ in range(40):s,r,d,tr,i=e.step(np.zeros(5))
print(seed,bx,speed,'right',sum(g(n,'x')>2 for n in names),'high',sum(g(n,'y')>1.5 for n in names),'done',d)
e.close()
