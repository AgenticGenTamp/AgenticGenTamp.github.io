from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1)
R = env.observation_space.get_type('mujoco_tidybot_robot')
r = obs.get_objects(R)[0]
f = ['pos_base_x','pos_base_y','pos_base_rot','pos_arm_joint1','pos_arm_joint2','pos_arm_joint4','pos_gripper']
def show(o, rew): print([round(o.get(r,k),3) for k in f], round(rew,3))
show(obs,0)
a = np.zeros(11); a[0]=0.1; a[3]=0.1
for i in range(5):
    obs, rew, te, tr, inf = env.step(a); show(obs, rew)
a = np.zeros(11); 
for i in range(3):
    obs, rew, te, tr, inf = env.step(a); show(obs, rew)
a[10]=1.0
for i in range(6):
    obs, rew, te, tr, inf = env.step(a); show(obs, rew)
a[10]=0.5
for i in range(4):
    obs, rew, te, tr, inf = env.step(a); show(obs, rew)
a[:]=0; a[2]=0.1
for i in range(3):
    obs, rew, te, tr, inf = env.step(a); show(obs, rew)
a[:]=0; a[0]=0.1
for i in range(3):
    obs, rew, te, tr, inf = env.step(a); show(obs, rew)
env.close()
