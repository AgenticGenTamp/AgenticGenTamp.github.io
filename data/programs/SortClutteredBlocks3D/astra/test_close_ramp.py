from env_client import make_env
from approach import GeneratedApproach
import time
class Ramped(GeneratedApproach):
 def get_action(self,s):
  a=super().get_action(s)
  if self.stage==2:a[-1]=min(1.,(self.age+1)/3.)
  return a
for seed in [0,1]:
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':20});p=Ramped(e.action_space,e.observation_space,{});p.reset(s,info);last=0;t=time.time()
 for i in range(1000):
  s,r,d,tr,_=e.step(p.get_action(s));n=sum(p.done(s,o) for o in p.cubes)
  if n!=last:print(seed,i,'DONE',n,flush=True);last=n
  if d or tr:break
 print('RESULT',seed,i+1,d,n,'TIME',time.time()-t,flush=True);e.close()
