from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
def rb(o):
    r=o.get_object_from_name('robot')
    return [round(float(o.get(r,f)),4) for f in ['pos_base_x','pos_base_y','pos_base_rot','pos_arm_joint1','pos_arm_joint2','pos_gripper']]
print("init", rb(obs))
a = np.zeros(11, dtype=np.float32); a[0]=0.1
for i in range(20):
    obs, rew, term, trunc, info = env.step(a)
    print(i, round(rew,4), rb(obs))
