from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
R = obs.get_object_from_name('robot')
for g in [1,1,1,-1,-1,-1,0.7,-0.7,0,1,0,-1]:
    a=np.zeros(11,dtype=np.float32); a[10]=g
    obs,r,te,tr,inf=env.step(a)
    print(g, obs.get(R,'finger_state'), obs.get(R,'grasp_active'), inf)
