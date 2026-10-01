from env_client import make_env
import numpy as np
np.set_printoptions(precision=3, suppress=True)
env = make_env()
for s in range(20):
    obs,_ = env.reset(seed=s)
    print(s, obs[0:3], obs[16:18], obs[29:32], obs[12:15])
