from env_client import make_env
from approach import GeneratedApproach
import numpy as np,time,sys
E=make_env();s,i=E.reset(seed=int(sys.argv[1]) if len(sys.argv)>1 else 0);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,i);old=-1;t0=time.monotonic()
for k in range(1000):
 s,r,t,tr,i=E.step(p.get_action(s))
 if p.stage!=old:
  old=p.stage;print(k,p.stage,r,'scoop',p.transform(s,'scoop_0')[:3,3].round(4),'quat',np.round(p.transform(s,'scoop_0')[:3,:3],2).tolist(),'green',sum(s.get(o,'y')>0 for o in p.cubes),flush=True)
 if t or tr:break
print('END',k,r,t,tr,time.monotonic()-t0,flush=True);E.close()
