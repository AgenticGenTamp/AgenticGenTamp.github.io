import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/Rearrange3D-o2-put_the_boxed_drink_and_the_can_next_to_the_bowl-v0")
re = env._kinder_env._object_centric_env._robot_env
for tgt in ([-0.236,-0.25,0.0],[-0.10,0.0,0.0],[-0.236,0.0,0.5]):
    env.reset(seed=0); tr=[]
    for k in range(12):
        b=np.array(re.qpos["base"]); a=np.zeros(11); a[:3]=np.clip(np.array(tgt)-b,-0.1,0.1); env.step(a); tr.append(np.round(np.array(re.qpos["base"]),3).tolist())
    print("target",tgt,"traj",tr[0],tr[2],tr[5],tr[-1])
