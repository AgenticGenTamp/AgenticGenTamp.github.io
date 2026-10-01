from probe_low_shelf import Low
from env_client import make_env
import numpy as np
class MultiLow(Low):
 def plan_place(self,s):
  original=self.fixtures
  self.targetfixture=original[self.index%len(original)]
  self.fixtures=[self.targetfixture]
  super().plan_place(s)
  self.fixtures=original
 def get_action(self,s):
  if self.stage==2 and len(self.queue)==3:
   p=self.xyz(s,self.objects[self.index]);target=self.xyz(s,self.targetfixture)
   if p[0]>target[0]-.1 and abs(p[1]-target[1])<.035 and self.waysteps>=8:self.waysteps=111
  return super().get_action(s)
e=make_env();s,info=e.reset(seed=0,options={'object_count':3})
a=MultiLow(e.action_space,e.observation_space,{});a.z=.30;a.reset(s,info);last=None
for t in range(1000):
 s,r,te,tr,_=e.step(a.get_action(s));key=(a.index,a.stage,len(a.queue))
 if key!=last or te:
  print('STEP',t,key,'pos',[np.round(a.xyz(s,o),4).tolist() for o in a.objects],'term',te,'trunc',tr,flush=True);last=key
 if te or tr or a.stage==3:break
print('FINAL',t,te,tr,flush=True);e.close()
