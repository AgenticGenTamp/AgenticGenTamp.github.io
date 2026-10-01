from env_client import make_env
from approach import GeneratedApproach
import numpy as np
seed=5;rng=np.random.default_rng(3);e=make_env();s,i=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
while p.stage<5:s,*_=e.step(p.get_action(s))
base=p.target(p.blocker)
def goto(q,grip=1):
 global s
 for k in range(12):
  a=p.motion(s,base,grip)
  for j in range(7):
   d=q[j]-p.g(s,'robot','joint_'+str(j+1));d=p.w(d) if j in (4,6) else d;a[3+j]=np.clip(d,-.2,.2)
  s,*_=e.step(a)
for trial in range(100):
 q=p.Q+rng.normal(0,[.18,.18,.25,.18,.25,.18,.25])
 goto(q,1);a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 if p.g(s,'robot','grasp_active'):
  print('HIT',trial,'base',p.robot(s),'q',[p.g(s,'robot','joint_'+str(j+1)) for j in range(7)]);break
else:print('none')
e.close()
