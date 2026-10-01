from env_client import make_env
from approach import GeneratedApproach
import numpy as np,time
class Guarded(GeneratedApproach):
 def next_stage(self,s):
  super().next_stage(s)
  if self.stage==4 and any(o!=self.obj and self.xyz(s,o)[2]>.48 and np.linalg.norm(self.targets[o.name][:2]-self.targets[self.obj.name][:2])>.06 for o in self.cubes):
   self.stage=6;self.grip=0
E=make_env();s,info=E.reset(seed=1,options={'object_count':20});p=Guarded(E.action_space,E.observation_space,{});p.reset(s,info);last=0;policy_time=0;t=time.time()
for i in range(1000):
 t0=time.time();a=p.get_action(s);policy_time+=time.time()-t0;s,r,d,tr,_=E.step(a);n=sum(p.done(s,o) for o in p.cubes)
 if n!=last:print(i,n,flush=True);last=n
 if d or tr:break
print('RESULT',i+1,d,n,'POLICY_SECONDS',policy_time,'TOTAL_SECONDS',time.time()-t,flush=True);E.close()
