from env_client import make_env
import numpy as np
POS=["pos_base_x","pos_base_y","pos_base_rot"]+[f"pos_arm_joint{i}" for i in range(1,8)]+["pos_gripper"]
VEL=["vel_base_x","vel_base_y","vel_base_rot"]+[f"vel_arm_joint{i}" for i in range(1,8)]+["vel_gripper"]
def rd(env,obs,F):
    r=obs.get_objects(env.observation_space.get_type("mujoco_tidybot_robot"))[0]
    return np.array([float(obs.get(r,f)) for f in F])
env=make_env(); obs,_=env.reset(seed=0)
print("INIT pos",np.round(rd(env,obs,POS),4))
prev=rd(env,obs,POS)
a=np.zeros(11,dtype=np.float32); a[0]=0.1
for i in range(4):
    obs,_,_,_,_=env.step(a); c=rd(env,obs,POS)
    print("step",i,"dx=%.5f"%(c[0]-prev[0]),"vel=",np.round(rd(env,obs,VEL)[:3],4)); prev=c
env.close()
