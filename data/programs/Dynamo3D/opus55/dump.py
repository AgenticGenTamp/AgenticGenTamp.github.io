from env_client import make_env
import numpy as np
env = make_env()
R = env.observation_space.get_type('mujoco_tidybot_robot')
M = env.observation_space.get_type('mujoco_movable_object')
for seed in range(20):
    obs, info = env.reset(seed=seed)
    r=obs.get_objects(R)[0]
    cs=sorted([(c.name,round(obs.get(c,'x'),2),round(obs.get(c,'y'),2),round(obs.get(c,'z'),2), round(2*np.arctan2(obs.get(c,'qz'),obs.get(c,'qw')),2)) for c in obs.get_objects(M)])
    print(seed, info['object_count'], [round(obs.get(r,f),2) for f in ['pos_base_x','pos_base_y','pos_base_rot']], cs)
