from env_client import make_env
from approach import GeneratedApproach
import numpy as np
class LiftRecovery(GeneratedApproach):
 def get_action(self,s):
  if self.stuck>=2 and self.stage==1 and s.get(self.r,'grasp_active'):
   p=np.array([s.get(self.r,f) for f in self.fs]);p[4]-=.10
   self.queue.insert(0,(p,0));self.stuck=0
  return super().get_action(s)
for seed in [54,46]:
 e=make_env();s,i=e.reset(seed=seed);a=LiftRecovery(e.action_space,e.observation_space,{});a.reset(s,i)
 for n in range(300):
  ac=a.get_action(s);s,_,t,tr,_=e.step(ac)
  if t or tr:break
 print('RESULT',seed,n+1,t,a.stage,s.get(a.r,'grasp_active'),flush=True);e.close()
