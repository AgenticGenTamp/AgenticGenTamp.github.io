from env_client import make_env
import numpy as np

env = make_env()
for seed in [0, 2, 5, 9, 12, 13]:
    o, _ = env.reset(seed=seed)
    a = np.zeros(11, np.float32)
    o, r, t, tr, _ = env.step(a)
    print(seed, r, t, tr, np.linalg.norm(o[16:18]-o[:2]), np.linalg.norm(o[32:34]-o[:2]))
env.close()
