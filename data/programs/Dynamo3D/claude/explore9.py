import numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=1)
# try arm joints and gripper effects; check reward variety
a=np.zeros(11,dtype=np.float32)
r=obs.get_object_from_name('robot')
tf=['pos_base_x','pos_base_y','pos_base_rot']+['pos_arm_joint%d'%i for i in range(1,8)]+['pos_gripper']
print("init",[round(float(obs.get(r,f)),3) for f in tf])
for k in range(3,11):
    a=np.zeros(11,dtype=np.float32); a[k]=0.1 if k<10 else 1.0
    for _ in range(5):
        obs,rew,term,trunc,info=env.step(a)
    r=obs.get_object_from_name('robot')
    print(k, round(rew,4), term, [round(float(obs.get(r,f)),3) for f in tf])
