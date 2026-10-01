from env_client import make_env
import numpy as np
with make_env() as e:
 for n in [1,3,12]:
  s,info=e.reset(seed=0, options={'object_count':n})
  print('COUNT',n,info,flush=True)
  for o in s.get_objects(e.observation_space.get_type('mujoco_movable_object')):
   print(o.name, *(round(s.get(o,f),3) for f in ['x','y','z']),flush=True)
