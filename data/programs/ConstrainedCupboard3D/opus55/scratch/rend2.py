import sys; sys.path.insert(0,'.')
from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = env.observation_space.get_type('mujoco_tidybot_robot'); M = env.observation_space.get_type('mujoco_movable_object')
r = obs.get_objects(R)[0]; s = obs.copy(); s.set(r,'pos_base_x',-1.5)
rods = sorted(s.get_objects(M), key=lambda o:o.name)
# vertical markers at y=0.33 (left of cupboard), x=1.9: z centers 0.15, 0.45 ; and one at y=-0.33 z=0.75
for rd,(x,y,z) in zip(rods, [(1.95,0.33,0.15),(1.95,0.33,0.45),(1.95,-0.33,0.75)]):
    for k,v in zip(['x','y','z','qw','qx','qy','qz'],[x,y,z,0.7071,0.7071,0,0]): s.set(rd,k,v)
print(env.render_state(state=s, label='markers'))
env.close()
