from env_client import make_env
from grasp_calibration_snapshot import GeneratedApproach
import numpy as np
import time
for obj in [0,54,70]:
 for seed in range(6):
  e=make_env();s,info=e.reset(seed=seed)
  p=GeneratedApproach(e.action_space,e.observation_space,{})
  p.mount=.4;p.close=1.;p.reset(s,info)
  p.b=np.array([s[obj]-.55-.14,s[obj+1],0.])
  initial=s[obj:obj+3].copy();mz=initial[2];start=time.monotonic()
  for k in range(300):
   s,r,t,tr,info=e.step(p.get_action(s));mz=max(mz,s[obj+2])
   if p.phase==3 and p.age>65:break
   if t or tr:break
  print('GRASP',obj,seed,'initial',np.round(initial,5),'maxz',round(mz,5),'last',np.round(s[obj:obj+3],5),'phase',p.phase,'steps',k+1,'seconds',round(time.monotonic()-start,2),flush=True)
  e.close()
