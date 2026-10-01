from env_client import make_env
import numpy as np
env=make_env();s,_=env.reset(seed=0,options={"object_count":1})
ro=s.get_object_from_name('robot')
for cmd in [1,1,-1,-1,1,0,-1]:
 a=np.zeros(11,np.float32);a[10]=cmd
 s,*rest=env.step(a);ro=s.get_object_from_name('robot')
 print(cmd,s.get(ro,'finger_state'),s.get(ro,'grasp_active'))
env.close()
