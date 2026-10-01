from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
R = env.observation_space.get_type('mujoco_tidybot_robot')
M = env.observation_space.get_type('mujoco_movable_object')
def rs(o):
    r=o.get_objects(R)[0]; return [round(o.get(r,f),3) for f in ['pos_base_x','pos_base_y','pos_base_rot']]
for t in range(5):
    a=np.zeros(11,dtype=np.float32)
    obs,rew,term,trunc,info=env.step(a); print(t,rs(obs),rew,term,trunc,info)
for t in range(5):
    a=np.zeros(11,dtype=np.float32); a[0]=0.1
    obs,rew,term,trunc,info=env.step(a); print('x',t,rs(obs),rew)
for t in range(5):
    a=np.zeros(11,dtype=np.float32); a[1]=0.1
    obs,rew,term,trunc,info=env.step(a); print('y',t,rs(obs),rew)
for t in range(5):
    a=np.zeros(11,dtype=np.float32); a[2]=0.1
    obs,rew,term,trunc,info=env.step(a); print('th',t,rs(obs),rew)
