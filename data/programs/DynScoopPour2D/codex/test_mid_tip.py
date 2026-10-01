import math, sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

target_y=float(sys.argv[1]); lift_speed=float(sys.argv[2]); seed=int(sys.argv[3])
e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
names=[n for n in s.get_object_names() if n.startswith('small_')]
g=lambda n,f:float(s.get(s.get_object_from_name(n),f))
while p.stage<8:s,*_=e.step(p.get_action(s))
for _ in range(160):
 s,r,d,tr,i=e.step(np.array([0,np.clip(target_y-g('robot','y'),-lift_speed,lift_speed),0,0,0]))
 if abs(target_y-g('robot','y'))<.01:break
for _ in range(30):
 er=(0-g('robot','theta')+math.pi)%(2*math.pi)-math.pi
 s,r,d,tr,i=e.step(np.array([0,0,np.clip(er,-.098,.098),0,0]))
 if d:break
for _ in range(25):s,r,d,tr,i=e.step(np.zeros(5))
print(target_y,lift_speed,'right',sum(g(n,'x')>1.75 for n in names),'done',d,'held',g('hook','held'),'robot_y',g('robot','y'))
e.close()
