from env_client import make_env
import numpy as np
POS=["pos_base_x","pos_base_y","pos_base_rot"]+[f"pos_arm_joint{i}" for i in range(1,8)]+["pos_gripper"]
def rd(env,obs):
    r=obs.get_objects(env.observation_space.get_type("mujoco_tidybot_robot"))[0]
    return np.array([float(obs.get(r,f)) for f in POS])
for idx in range(10):
    env=make_env(); obs,_=env.reset(seed=0); p0=rd(env,obs)
    a=np.zeros(11,dtype=np.float32); a[idx]=0.1
    obs,_,_,_,_=env.step(a); p1=rd(env,obs)
    obs,_,_,_,_=env.step(a); p2=rd(env,obs)
    d=p2-p1
    big=[(POS[j],round(float(d[j]),4)) for j in range(11) if abs(d[j])>1e-3]
    print("act[%d]=0.1 -> per-step:"%idx, big)
    env.close()
