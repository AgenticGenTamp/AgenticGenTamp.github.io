from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
r=obs.get_object_from_name('robot')
F=['pos_base_x','pos_base_y','pos_base_rot','pos_arm_joint1','pos_arm_joint2','pos_arm_joint3','pos_arm_joint4','pos_arm_joint5','pos_arm_joint6','pos_arm_joint7','pos_gripper']
def rs(o): return np.array([o.get(r,f) for f in F]).round(3)
print(rs(obs))
a=np.zeros(11,dtype=np.float32)
for t in range(3):
    obs,rew,te,tr,info=env.step(a); print('zero',rs(obs),rew,info)
a[0]=0.1
for t in range(3):
    obs,rew,te,tr,info=env.step(a); print('bx',rs(obs),rew)
a[:]=0; a[2]=0.1
for t in range(3):
    obs,rew,te,tr,info=env.step(a); print('brot',rs(obs),rew)
a[:]=0; a[0]=0.1
for t in range(3):
    obs,rew,te,tr,info=env.step(a); print('bx after rot',rs(obs),rew)
a[:]=0; a[3]=0.1
for t in range(3):
    obs,rew,te,tr,info=env.step(a); print('j1',rs(obs),rew)
a[:]=0; a[10]=1
for t in range(5):
    obs,rew,te,tr,info=env.step(a); print('grip',rs(obs),rew)
print(dir(env))
