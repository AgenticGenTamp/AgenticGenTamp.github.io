from env_client import make_env
import numpy as np

env = make_env()
for seed in range(5, 15):
    o, _ = env.reset(seed=seed)
    p = o[[0,1,2,16,17,18,32,33,34]].reshape(3,3)
    bb = o[[13,14,15,29,30,31,45,46,47]].reshape(3,3)
    d = np.linalg.norm(p[1:,:2] - p[0,:2], axis=1)
    print(seed, 'p', np.round(p,3).tolist(), 'bb', np.round(bb,3).tolist(), 'dxy', np.round(d,3).tolist())
env.close()
