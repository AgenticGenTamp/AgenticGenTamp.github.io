from env_client import make_env
import numpy as np
env = make_env()
M = env.observation_space.get_type('mujoco_movable_object')
R = env.observation_space.get_type('mujoco_tidybot_robot')
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed, options={'object_count':12})
    r=obs.get_objects(R)[0]
    print(seed, [round(obs.get(r,f),2) for f in ['pos_base_x','pos_base_y','pos_base_rot']])
    for c in sorted(obs.get_objects(M),key=lambda c:c.name): print('  ',c.name,[round(obs.get(c,f),2) for f in ['x','y','z','bb_x','bb_y','bb_z']])
