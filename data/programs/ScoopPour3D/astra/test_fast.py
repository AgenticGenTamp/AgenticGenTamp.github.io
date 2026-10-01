from env_client import make_env
from approach import GeneratedApproach
import sys,time
E=make_env();s,i=E.reset(seed=int(sys.argv[1]) if len(sys.argv)>1 else 0,options={'object_count':int(sys.argv[2])} if len(sys.argv)>2 else None);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,i);old='';t0=time.monotonic()
for k in range(1000):
 s,r,t,tr,i=E.step(p.get_action(s))
 if p.stage!=old:
  old=p.stage
  print(k,p.stage,p.target.name if p.target else '',r,'green',sum(s.get(o,'y')>0 for o in p.cubes),'base',p.robot(s)[:2].round(3),flush=True)
 if t or tr:break
print('END',k,r,t,tr,time.monotonic()-t0,flush=True);E.close()
