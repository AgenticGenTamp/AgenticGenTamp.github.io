from env_client import make_env
import numpy as np
env = make_env()
os_ = env.observation_space
T = os_.get_type
for s in range(8):
    obs, info = env.reset(seed=s)
    mo = obs.get_objects(T("mujoco_movable_object"))
    r = obs.get_objects(T("mujoco_tidybot_robot"))[0]
    print("seed",s,"n",info.get("object_count"))
    print("  rob", [round(float(obs.get(r,f)),3) for f in ["pos_base_x","pos_base_y","pos_base_rot"]],
          "arm",[round(float(obs.get(r,'pos_arm_joint%d'%i)),3) for i in range(1,8)])
    for o in mo:
        print("   ",o.name,[round(float(obs.get(o,f)),3) for f in ["x","y","z","qw","qz","bb_x","bb_y","bb_z"]])
env.close()
