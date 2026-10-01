from env_client import make_env
from grasp_calibration_snapshot import GeneratedApproach
import numpy as np
import time
import sys
mount=float(sys.argv[1]) if len(sys.argv)>1 else .36
dx=float(sys.argv[2]) if len(sys.argv)>2 else -.11
pairs=[(0,3),(0,5),(54,5),(70,0)]
pairs += [(o,s) for o in [0,54,70] for s in range(6) if (o,s) not in pairs]
for obj,seed in pairs:
 e=make_env();s,info=e.reset(seed=seed)
 p=GeneratedApproach(e.action_space,e.observation_space,{})
 p.mount=mount;p.close=1.;p.reset(s,info)
 p.b=np.array([s[obj]-.55+dx,s[obj+1],0.])
 initial=s[obj:obj+3].copy();mz=initial[2];start=time.monotonic();ages=[];lastphase=0
 for k in range(330):
  oldage=p.age
  s,r,t,tr,info=e.step(p.get_action(s));mz=max(mz,s[obj+2])
  if p.phase!=lastphase:ages.append(oldage+1);lastphase=p.phase
  if p.phase==3 and p.age>65:break
  if t or tr:break
 print('HIGH',mount,dx,obj,seed,'maxz',round(mz,5),'last',np.round(s[obj:obj+3],5),'phase',p.phase,'steps',k+1,'ages',ages,'seconds',round(time.monotonic()-start,2),flush=True)
 e.close()
