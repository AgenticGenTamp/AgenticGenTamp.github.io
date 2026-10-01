from env_client import make_env
import numpy as np
FE=["pos_base_x","pos_base_y","pos_base_rot","pos_arm_joint1","pos_arm_joint2","pos_arm_joint3","pos_arm_joint4","pos_arm_joint5","pos_arm_joint6","pos_arm_joint7","pos_gripper"]
def rd(env,obs):
    r=obs.get_objects(env.observation_space.get_type("mujoco_tidybot_robot"))[0]
    return np.array([float(obs.get(r,f)) for f in FE])
env=make_env(); obs,_=env.reset(seed=0)
prev=rd(env,obs); print("init",np.round(prev,4))
a=np.zeros(11,dtype=np.float32); a[0]=0.1
for i in range(5):
    obs,_,_,_,_=env.step(a); c=rd(env,obs); print("d",i,np.round(c-prev,4)); prev=c
z=np.zeros(11,dtype=np.float32)
for i in range(4):
    obs,_,_,_,_=env.step(z); c=rd(env,obs); print("z",i,np.round(c-prev,5)); prev=c
# half command
a2=np.zeros(11,dtype=np.float32); a2[0]=0.05
for i in range(3):
    obs,_,_,_,_=env.step(a2); c=rd(env,obs); print("h",i,np.round(c-prev,4)); prev=c
# negative
a3=np.zeros(11,dtype=np.float32); a3[0]=-0.1
for i in range(3):
    obs,_,_,_,_=env.step(a3); c=rd(env,obs); print("n",i,np.round(c-prev,4)); prev=c
env.close()
