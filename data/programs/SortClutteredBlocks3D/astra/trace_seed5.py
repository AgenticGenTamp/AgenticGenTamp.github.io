from env_client import make_env
from approach import GeneratedApproach
import numpy as np
E=make_env();s,info=E.reset(seed=5);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info);prev=None
for i in range(430):
 key=(p.obj.name if p.obj else None,p.stage)
 if key!=prev:
  print('STAGE',i,key,'base',np.round(p.base,3),'z',p.z,'pos',{o.name:p.xyz(s,o).round(3).tolist() for o in p.cubes},flush=True);prev=key
 a=p.get_action(s);s,r,d,tr,_=E.step(a)
 if d or tr:print('DONE',i,d,flush=True);break
E.close()
