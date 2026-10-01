import sys
from env_client import make_env
n=int(sys.argv[1]); seed=int(sys.argv[2])
e=make_env(); s,i=e.reset(seed=seed,options={'object_count':n})
r=s.get_object_from_name('robot'); print(i,'robot',float(s.get(r,'pos_base_x')),float(s.get(r,'pos_base_y')))
for o in s.get_objects(e.observation_space.get_type('mujoco_movable_object')): print(o.name,float(s.get(o,'x')),float(s.get(o,'y')))
e.close()
