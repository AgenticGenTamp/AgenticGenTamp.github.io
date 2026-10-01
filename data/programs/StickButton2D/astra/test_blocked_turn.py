from env_client import make_env
from approach import GeneratedApproach, wrap
import math,numpy as np
E=make_env();s,info=E.reset(seed=0,options={'object_count':20})
p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info)
stage=0;goal=None;seen_recovery=False
for k in range(500):
 r=s.get_object_from_name('robot');st=s.get_object_from_name('stick')
 if p.phase=='held':
  if stage==0 or (stage==1 and abs(wrap(s.get(st,'theta')-math.radians(130)))<.02):
   alpha=math.radians(130 if stage==0 else -130)
   t=wrap(s.get(r,'theta')+alpha-s.get(st,'theta'))
   p.goal=np.array([1.75,1.14,t,.2]);stage+=1
   print('FORCED',stage,p.goal,flush=True)
  if stage<3 and p.recovery is not None:
   seen_recovery=True;stage=3
   print('RECOVERY',k,p.recovery,flush=True)
 a=p.get_action(s);s,rw,done,trunc,_=E.step(a)
 if done or trunc:break
print('RESULT',done,k+1,'seen_recovery',seen_recovery,'stage',stage)
E.close()
