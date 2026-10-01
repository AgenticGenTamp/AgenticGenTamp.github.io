import sys
import numpy as np
from env_client import make_env
n=int(sys.argv[1]); seed=int(sys.argv[2]); e=make_env(); s,i=e.reset(seed=seed,options={'object_count':n}); r=s.get_object_from_name('robot')
chairs=s.get_objects(e.observation_space.get_type('mujoco_movable_object'))
tx=round(max(float(s.get(o,'x')) for o in chairs)); ty=round(max(float(s.get(o,'y')) for o in chairs)); print('target',tx,ty,flush=True)
for k in range(120):
 x=float(s.get(r,'pos_base_x')); y=float(s.get(r,'pos_base_y')); a=np.zeros(11,np.float32); a[0]=np.clip((tx-x)*.2,-.1,.1); a[1]=np.clip((ty-y)*.2,-.1,.1)
 s,rw,t,tr,_=e.step(a)
 if k%10==9 or t or tr: print(k+1,rw,t,tr,round(float(s.get(r,'pos_base_x')),2),round(float(s.get(r,'pos_base_y')),2),flush=True)
 if t or tr: break
e.close()
