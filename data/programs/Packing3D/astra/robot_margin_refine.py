from env_client import make_env
from approach import GeneratedApproach
import numpy as np
class Margin(GeneratedApproach):
 def get_action(self,s):
  a=super().get_action(s)
  if hasattr(self,'drop'):self.drop=np.array([self.margin,0,.1])
  return a
E=make_env()
for x in [.265,.27,.275,.28,.32,.33,.335]:
 s,info=E.reset(seed=0,options={'object_count':2});A=Margin(E.action_space,E.observation_space,{});A.reset(s,info);A.margin=x;stall=0
 for k in range(60):
  a=A.get_action(s);ns,r,t,tr,info=E.step(a)
  part=ns.get_object_from_name('part0')
  if A.phase=='lower' and np.linalg.norm(A.xyz(ns,part)-A.xyz(s,part))<1e-7:stall+=1
  else:stall=0
  s=ns
  if (A.phase=='release' and s.get(part,'grasp_active')<.5) or stall>=3:
   print('X',x,'step',k,'phase',A.phase,'pose',A.xyz(s,part),'held',s.get(part,'grasp_active'),'stall',stall,flush=True);break
 else:print('TIMEOUT',x,A.phase,flush=True)
E.close()
