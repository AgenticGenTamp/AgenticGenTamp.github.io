from env_client import make_env
import numpy as np, sys
env = make_env()
obs, info = env.reset(seed=0)
R = env.observation_space.get_type('mujoco_tidybot_robot')
M = env.observation_space.get_type('mujoco_movable_object')
r = obs.get_objects(R)[0]
s = obs.copy()
s.set(r,'pos_base_x',-1.5)
rods = s.get_objects(M)
# rod vertical at 2.0, 0.05 ; rod along x
s.set(rods[0],'x',1.9); s.set(rods[0],'y',0.05); s.set(rods[0],'z',0.5); s.set(rods[0],'qw',0.7071); s.set(rods[0],'qx',0.7071); s.set(rods[0],'qy',0); s.set(rods[0],'qz',0)
s.set(rods[1],'x',1.5); s.set(rods[1],'y',-0.25); s.set(rods[1],'z',0.3); s.set(rods[1],'qw',0.7071); s.set(rods[1],'qx',0); s.set(rods[1],'qy',0); s.set(rods[1],'qz',0.7071)
p = env.render_state(state=s, label='test')
print(p)
env.close()
