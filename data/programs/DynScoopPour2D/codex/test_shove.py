import math,sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
tt=float(sys.argv[1]);joint=float(sys.argv[2]);seed=int(sys.argv[3]);e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
names=[n for n in s.get_object_names() if n.startswith('small_')];g=lambda n,f:float(s.get(s.get_object_from_name(n),f))
while p.stage<8:s,*_=e.step(p.get_action(s))
for _ in range(40):
 er=(tt-g('robot','theta')+math.pi)%(2*math.pi)-math.pi
 s,r,d,tr,i=e.step(np.array([0,0,np.clip(er,-.03,.03),np.clip(joint-g('robot','arm_joint'),-.02,.02),0]))
 if d:break
for _ in range(30):s,r,d,tr,i=e.step(np.zeros(5))
print(tt,joint,'right',sum(g(n,'x')>1.75 for n in names),'held',g('hook','held'),'hook',g('hook','x'),g('hook','y'),'theta',g('robot','theta'))
e.close()
