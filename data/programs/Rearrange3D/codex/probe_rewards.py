from env_client import make_env
import numpy as np


def summary(seed):
    env = make_env()
    obs, info = env.reset(seed=seed)
    pts = obs[[0,1,2,16,17,18,32,33,34,48,49,50,64,65,66,77,78,79,84,85,86,93,94,95,103]]
    a = np.zeros(11, dtype=np.float32)
    a[10] = obs[103]
    obs2, rew, term, trunc, inf = env.step(a)
    print(seed, 'max', env.max_steps, 'pts', np.round(pts,3).tolist(), 'r0', rew, term, trunc, 'info', info, inf)
    env.close()


for seed in range(20):
    summary(seed)
