from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
r=obs.get_object_from_name('robot')
F=['pos_base_x','pos_base_y','pos_base_rot','pos_arm_joint1','pos_arm_joint2','pos_arm_joint3','pos_arm_joint4','pos_arm_joint5','pos_arm_joint6','pos_arm_joint7','pos_gripper']
def rs(o): return np.array([o.get(r,f) for f in F]).round(3)
a=np.zeros(11,dtype=np.float32)
for j in [3,4,5,6,7,8,9]:
    a[:]=0; a[j]=0.1
    out=[]
    for t in range(6):
        obs,rew,te,tr,info=env.step(a); out.append(rs(obs)[j])
    a[:]=0
    for t in range(4):
        obs,rew,te,tr,info=env.step(a); out.append(rs(obs)[j])
    print(j, out)
