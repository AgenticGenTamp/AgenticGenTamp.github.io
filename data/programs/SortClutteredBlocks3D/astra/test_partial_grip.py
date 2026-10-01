from env_client import make_env
from approach import GeneratedApproach
import numpy as np
import time
class Partial(GeneratedApproach):
 def reset(self,s,info):
  self.liftfails=0;self.lifts=0;self.multilifts=0;self.graspbefore={};super().reset(s,info)
 def next_stage(self,s):
  if self.stage==1:self.graspbefore={o.name:self.xyz(s,o)[2] for o in self.cubes}
  if self.stage==3:
   if self.xyz(s,self.obj)[2]<.475:self.liftfails+=1
   else:
    self.lifts+=1
    raised=[o.name for o in self.cubes if self.xyz(s,o)[2]>self.graspbefore.get(o.name,1)+.04]
    if len(raised)>1:self.multilifts+=1
  super().next_stage(s)
 def get_action(self,s):
  a=super().get_action(s)
  if self.stage in (0,1):a[-1]=self.partial
  return a
for partial in [0,.25,.4,.6]:
 e=make_env();s,info=e.reset(seed=0,options={'object_count':20});p=Partial(e.action_space,e.observation_space,{});p.partial=partial;p.reset(s,info)
 bins=[s.get_object_from_name('bin_'+c) for c in ['red','green','blue','yellow']];bs={o.name:p.xyz(s,o).copy() for o in bins};maxbin=0;start=time.time();peak=0
 for step in range(400):
  s,r,d,tr,_=e.step(p.get_action(s));n=sum(p.done(s,o) for o in p.cubes);peak=max(peak,n)
  maxbin=max(maxbin,max(np.linalg.norm(p.xyz(s,o)-bs[o.name]) for o in bins))
  if d or tr:break
 print('RESULT partial',partial,'sorted',n,'peak',peak,'liftfails',p.liftfails,'lifts',p.lifts,'multi',p.multilifts,'maxbin',round(maxbin,4),'seconds',round(time.time()-start,2),'gripper',s.get(p.robot,'pos_gripper'),flush=True)
 e.close()
