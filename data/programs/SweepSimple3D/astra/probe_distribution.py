from env_client import make_env
import numpy as np
E=make_env()
for seed in range(12):
 s,i=E.reset(seed=seed)
 print('SEED',seed,'INFO',i,flush=True)
 for name in ('mujoco_movable_object','mujoco_fixture','mujoco_tidybot_robot'):
  typ=E.observation_space.get_type(name)
  fs=['x','y','z'] if name!='mujoco_tidybot_robot' else ['pos_base_x','pos_base_y','pos_base_rot']
  print(name,[(str(o),[float(s.get(o,f)) for f in fs]) for o in s.get_objects(typ)],flush=True)
 for _ in range(3):
  s,r,t,tr,i=E.step(np.zeros(11,dtype=np.float32))
 print('ZERO',r,t,tr,i,flush=True)
E.close()
