from env_client import make_env
import numpy as np
env = make_env()
obs, info = env.reset(seed=1, options={'object_count':20})
print(info)
a = np.zeros(11, dtype=np.float32)
obs, rew, term, trunc, info = env.step(a)
print('20cubes rew', rew)
env.close()
