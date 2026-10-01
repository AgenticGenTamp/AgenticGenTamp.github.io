from env_client import make_env
from approach import GeneratedApproach
import numpy as np,time
class Rotated(GeneratedApproach):
 def ik(self,z):
  q=super().ik(z).copy();q[-1]+=np.pi/2;return q
for seed in [0,1]:
 e=make_env();s,info=e.reset(seed=seed,options={'object_count':20});p=Rotated(e.action_space,e.observation_space,{});p.reset(s,info);last=0;t=time.time()
 for i in range(1000):
  s,r,d,tr,_=e.step(p.get_action(s));n=sum(p.done(s,o) for o in p.cubes)
  if n!=last: print(seed,i,'DONE',n,flush=True);last=n
  if d or tr:break
 print('RESULT',seed,i+1,d,last,'SECONDS',time.time()-t,'ATTEMPTS',p.attempts,flush=True)
 e.close()
