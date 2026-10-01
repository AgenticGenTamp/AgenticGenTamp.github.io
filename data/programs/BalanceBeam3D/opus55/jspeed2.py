from env_client import make_env
import numpy as np
env = make_env()
obs, _ = env.reset(seed=0)
q0 = obs[19:26].copy()
for t in range(12):
    a = np.zeros(11, np.float32)
    if t < 3: a[3] = 0.1
    obs, *_ = env.step(a)
    print(t, "j1 moved %.4f vel %.3f" % (obs[19]-q0[0], obs[30]), "base", obs[16:18])
env.close()
