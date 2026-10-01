from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
R = obs.get_objects(env.observation_space.get_type('mujoco_tidybot_robot'))[0]
def rs(o): return [round(float(o.get(R,f)),3) for f in ['pos_base_x','pos_base_y','pos_base_rot','pos_arm_joint1','pos_arm_joint2','pos_gripper']]
for t in range(5):
    a = np.zeros(18, dtype=np.float32)
    a[0]=0.1
    obs, r, term, trunc, info = env.step(a)
    print(t, r, term, rs(obs))
env.close()
