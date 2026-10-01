from env_client import make_env
from approach_conditional import GeneratedApproach
import numpy as np
E=make_env();s,info=E.reset(seed=24);p=GeneratedApproach(E.action_space,E.observation_space,{});p.reset(s,info);bins=[s.get_object_from_name('bin_'+c) for c in ['red','green','blue','yellow']];initialbins={o.name:p.xyz(s,o).copy() for o in bins};first=False
for i in range(1000):
 s,r,d,tr,_=E.step(p.get_action(s));all_done=all(p.done(s,o) for o in p.cubes)
 if all_done and not first or i==999 or d:
  first=True;print('REPORT step',i,'terminated',d,'reward',r,flush=True)
  for o in bins:print('BIN',o.name,'initial',initialbins[o.name],'current',p.xyz(s,o),'quat',[s.get(o,f) for f in ['qw','qx','qy','qz']],flush=True)
  for o in p.cubes:
   b=bins[(int(o.name[4:])-1)%4];pos=p.xyz(s,o);target=p.targets[o.name]
   print('CUBE',o.name,'xyz',pos,'initial_delta',pos-target,'current_delta',pos-p.xyz(s,b),'dist3',np.linalg.norm(pos-target),'dist2',np.linalg.norm(pos[:2]-target[:2]),'quat',[s.get(o,f) for f in ['qw','qx','qy','qz']],flush=True)
 if d or tr:break
E.close()
