from env_client import make_env
from approach import GeneratedApproach
import numpy as np
for seed in [11,14,21,30,34,40,42,47]:
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 p.theta=0.;p.off=p.OFF.copy();t=p.target(p.blocker)
 for k in range(15):s,*_=e.step(p.motion(s,t,1))
 a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 print(seed,'target',np.round(t,3),'base',np.round(p.robot(s),3),'held',p.g(s,'robot','grasp_active'))
 e.close()
