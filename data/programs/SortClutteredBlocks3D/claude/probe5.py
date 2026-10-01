from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=0)
r=env.robot=None
for i in range(5):
    a = np.zeros(11, dtype=np.float32)
    obs, rew, term, trunc, info = env.step(a)
    o=obs.get_object_from_name('robot')
    print(i, rew, term, trunc, info, round(float(obs.get(o,'pos_base_x')),4), round(float(obs.get(o,'pos_gripper')),4))
env.close()
