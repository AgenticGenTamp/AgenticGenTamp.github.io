from env_client import make_env
import numpy as np
env = make_env()
obs, _ = env.reset(seed=0)
for j in range(7):
    for sgn in [1, -1]:
        q0 = obs[19:26].copy()
        for t in range(6):
            a = np.zeros(11, np.float32); a[3+j] = 0.1*sgn
            obs, *_ = env.step(a)
        d = obs[19:26] - q0
        print("joint", j+1, "sgn", sgn, "moved in 6 steps %.3f" % d[j], "vel %.3f" % obs[30+j])
env.close()
