import numpy as np
from env_client import make_env
env = make_env()
print("max_steps", env.max_steps)
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for s in range(5):
    obs, info = env.reset(seed=s)
    print(s, obs, info)
