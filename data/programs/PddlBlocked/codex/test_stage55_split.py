from env_client import make_env
from approach import GeneratedApproach
import numpy as np
for seed in [14,21,42,77]:
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 for k in range(100):
  s,*_=e.step(p.get_action(s))
  if p.stage==55:break
 q0=p.g(s,'robot','joint_4')
 for _ in range(3):
  a=np.zeros(11,np.float32);a[6]=np.clip(p.Q[3]+.20-p.g(s,'robot','joint_4'),-.2,.2);s,*_=e.step(a)
 t=p.target(p.blocker)+(p.green-p.blocker)+np.array([.015,-.1025])
 for _ in range(5):s,*_=e.step(p.motion(s,t,1,q4add=.20))
 a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 print(seed,'q4',q0,p.g(s,'robot','joint_4'),'base',p.robot(s),'target',t,'held',p.g(s,'robot','grasp_active'))
 e.close()
