from env_client import make_env
import numpy as np
env=make_env()
obs,_=env.reset(seed=0); r=obs.get_object_from_name('robot')
for v in [-1,-1,-1,1,1,-0.9,0.9]:
    a=np.zeros(11,dtype=np.float32); a[10]=v
    obs,*_=env.step(a); print(v, obs.get(r,'finger_state'), obs.get(r,'grasp_active'))
for v in [-1,1]:
    a=np.zeros(11,dtype=np.float32); a[10]=v; a[3]=0.05
    obs,*_=env.step(a); print('with motion',v, obs.get(r,'finger_state'), obs.get(r,'joint_1'))
env.close()
