from env_client import make_env
import numpy as np
FE=["pos_base_x","pos_base_y","pos_base_rot","pos_arm_joint1","pos_arm_joint2","pos_arm_joint3","pos_arm_joint4","pos_arm_joint5","pos_arm_joint6","pos_arm_joint7","pos_gripper"]
def rd(env,obs):
    r=obs.get_objects(env.observation_space.get_type("mujoco_tidybot_robot"))[0]
    return np.array([float(obs.get(r,f)) for f in FE])
env=make_env(); obs,_=env.reset(seed=0)
b=rd(env,obs)
print("base   ", np.round(b,4))
a=np.zeros(11,dtype=np.float32); a[0]=0.1
for i in range(10):
    obs,_,_,_,_=env.step(a); print("x+",i,np.round(rd(env,obs)-b,4))
z=np.zeros(11,dtype=np.float32)
p=rd(env,obs)
for i in range(6):
    obs,_,_,_,_=env.step(z); print("zero",i,np.round(rd(env,obs)-p,5))
env.close()
