from env_client import make_env
from approach import GeneratedApproach
import sys
seed=int(sys.argv[1]);e=make_env();s,info=e.reset(seed=seed);p=GeneratedApproach(e.action_space,e.observation_space,{});p.reset(s,info)
print('INITIAL',[(o.name,p.xyz(s,o).round(4).tolist()) for o in p.cubes],flush=True)
for i in range(500):
 old=(p.obj.name if p.obj else None,p.stage);a=p.get_action(s);s,r,d,tr,_=e.step(a);new=(p.obj.name if p.obj else None,p.stage)
 if old!=new:print(i,old,new,'ang',round(p.angle,2),'cube',[(o.name,p.xyz(s,o).round(3).tolist()) for o in p.cubes],flush=True)
 if d or tr: print('END',i,d,flush=True);break
e.close()
