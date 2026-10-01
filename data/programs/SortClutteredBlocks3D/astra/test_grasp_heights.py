from env_client import make_env
from approach import GeneratedApproach
import sys,time,numpy as np
for offset in [.005,.01,.015,.02]:
 class Raised(GeneratedApproach):
  def choose(self,state):
   super().choose(state)
   if self.obj is not None:self.pickz+=offset
 e=make_env();s,info=e.reset(seed=0,options={'object_count':20});p=Raised(e.action_space,e.observation_space,{});p.reset(s,info);peak=0;fails=0;lifts=0;t=time.time()
 for i in range(400):
  old=p.stage;a=p.get_action(s)
  if old==3:
   if p.stage==0:fails+=1
   if p.stage==4:lifts+=1
  s,r,d,tr,_=e.step(a);n=sum(p.done(s,o) for o in p.cubes);peak=max(peak,n)
  if d or tr:break
 print('RESULT',offset,i+1,d,'done',n,'peak',peak,'fails',fails,'lifts',lifts,'time',round(time.time()-t,1),flush=True);e.close()
