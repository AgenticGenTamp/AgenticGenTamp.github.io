from env_client import make_env
from narrow_pick import NarrowPick
import numpy as np
for gate in [.48,.44,.40,.36,.32]:
 e=make_env();s,info=e.reset(seed=112);p=NarrowPick(e.observation_space);p.reset(s)
 o=s.get_object_from_name('target_block');r=s.get_object_from_name('robot');caught=False
 for k in range(220):
  a=p.get_action(s)
  if p.phase=='descend' and s.get(r,'y')<s.get(o,'y')+s.get(o,'height')/2+gate:a[4]=-.019
  s,rew,t,tr,i=e.step(np.clip(a,e.action_space.low*.999,e.action_space.high*.999))
  if s.get(o,'held'):
   caught=True;break
  if k%50==0:print('progress',gate,k,p.phase,[round(s.get(o,f),3) for f in ['x','y','theta']],flush=True)
 print('RESULT',gate,k,caught,'target',[round(s.get(o,f),3) for f in ['x','y','theta']],'robot',[round(s.get(r,f),3) for f in ['x','y','theta','finger_gap']],flush=True)
 e.close()
