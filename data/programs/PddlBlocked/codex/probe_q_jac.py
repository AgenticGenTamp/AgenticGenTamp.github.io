from env_client import make_env
from approach import GeneratedApproach
import numpy as np
e=make_env()
for j in range(7):
 for sign in [-1,1]:
  s,i=e.reset(seed=27);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
  # run until blocker grasp
  for k in range(100):
   s,*_=e.step(p.get_action(s))
   if p.stage==2:break
  before=np.array([p.g(s,'blocker','pose_'+x) for x in 'xyz'])
  a=np.zeros(11,np.float32);a[3+j]=sign*.1;s,*_=e.step(a)
  after=np.array([p.g(s,'blocker','pose_'+x) for x in 'xyz'])
  print(j+1,sign,np.round(after-before,4),'q',p.g(s,'robot','joint_'+str(j+1)))
e.close()
