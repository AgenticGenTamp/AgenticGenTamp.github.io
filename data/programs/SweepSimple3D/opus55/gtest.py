import numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0)
R=obs.get_object_from_name('robot')
for g in [0.0,0.3,0.5,0.6,0.7,0.8,1.0]:
    a=np.zeros(11,np.float32); a[10]=g
    for _ in range(12): obs,*_=env.step(a)
    print(g, round(obs.get(R,'pos_gripper'),4))
