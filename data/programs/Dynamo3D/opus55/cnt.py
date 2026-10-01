from env_client import make_env
import numpy as np
env = make_env()
M = env.observation_space.get_type('mujoco_movable_object')
R = env.observation_space.get_type('mujoco_tidybot_robot')
for n in [1,2,3,4,5,6]:
  for seed in [0,1]:
    try:
        obs, info = env.reset(seed=seed, options={'object_count':n})
    except Exception as e:
        print(n, 'ERR', str(e)[:200]); continue
    r=obs.get_objects(R)[0]
    print(n, seed, info, [round(obs.get(r,f),2) for f in ['pos_base_x','pos_base_y']], sorted([(c.name,round(obs.get(c,'x'),2),round(obs.get(c,'y'),2)) for c in obs.get_objects(M)]))
