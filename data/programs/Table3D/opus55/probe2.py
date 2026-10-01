from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
r=obs.get_object_from_name('robot')
for i in range(8):
    a=np.zeros(11,dtype=np.float32); a[0]=0.1
    obs,*_=env.step(a); print('bx',round(obs.get(r,'pos_base_x'),3))
obs, info = env.reset(seed=0)
# lower arm with joint_2 forward
for i in range(20):
    a=np.zeros(11,dtype=np.float32); a[4]=0.1
    obs,*_=env.step(a); print('j2',round(obs.get(r,'joint_2'),3), [round(obs.get(c,'pose_z'),3) for c in obs.get_objects(env.observation_space.get_type('Kinematic3DCuboid'))])
env.close()
