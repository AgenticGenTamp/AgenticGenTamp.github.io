from env_client import make_env
from approach import GeneratedApproach
import numpy as np
X=np.array([.126276,.047549,-.012627,.241,-.042077,.838,-1.0])
for seed in [5,15,0]:
 e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
 while p.stage<5:s,*_=e.step(p.get_action(s))
 q=p.Q+X;t=p.target(p.blocker)
 for k in range(10):s,*_=e.step(p.motion(s,t,1,lift=True))
 for k in range(3):s,*_=e.step(p.motion(s,t,1))
 for k in range(20):
  a=p.motion(s,t,1)
  for j in range(7):
   d=q[j]-p.g(s,'robot','joint_'+str(j+1));d=p.w(d) if j in (4,6) else d;a[3+j]=np.clip(d,-.2,.2)
  s,*_=e.step(a)
 a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 print(seed,'base',p.robot(s),'q',[round(p.g(s,'robot','joint_'+str(j+1)),2) for j in range(7)],'held',p.g(s,'robot','grasp_active'))
 e.close()
