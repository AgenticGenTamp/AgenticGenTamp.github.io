from env_client import make_env
from approach import GeneratedApproach
import numpy as np

seed=23; rng=np.random.default_rng(11); e=make_env(); s,i=e.reset(seed=seed)
p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,i)
while p.stage<5:s,*_=e.step(p.get_action(s))
bt=p.target(p.blocker)
# return safely to the old blocker pose, then lower.
for _ in range(10):s,*_=e.step(p.motion(s,bt,1,lift=True))
for _ in range(3):s,*_=e.step(p.motion(s,bt,1))
print('ready',p.robot(s),bt)
for trial in range(300):
 q=p.Q+rng.normal(0,[.35,.18,.4,.3,.5,.5,.7])
 # Occasionally move the base as far inward as collision permits.
 frac=rng.uniform(0,.8); target=bt+frac*(p.target(p.green)-bt)
 for k in range(8):
  a=p.motion(s,target,1)
  for j in range(7):
   d=q[j]-p.g(s,'robot','joint_'+str(j+1));d=p.w(d) if j in (4,6) else d
   a[3+j]=np.clip(d,-.2,.2)
  s,*_=e.step(a)
 a=np.zeros(11,np.float32);a[10]=-1;s,*_=e.step(a)
 if p.g(s,'robot','grasp_active'):
  print('HIT',trial,'frac',frac,'base',p.robot(s),'q',[round(p.g(s,'robot','joint_'+str(j+1)),5) for j in range(7)]);break
 # Reset arm at the safe blocker pose via lifted configuration.
 for k in range(3):s,*_=e.step(p.motion(s,bt,1,lift=True))
 for k in range(3):s,*_=e.step(p.motion(s,bt,1))
else:print('none')
e.close()
