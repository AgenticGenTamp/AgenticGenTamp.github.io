from env_client import make_env
import numpy as np
np.set_printoptions(precision=3, suppress=True)
env = make_env()
print(env.max_steps)
for s in range(3):
    obs,_ = env.reset(seed=s)
    print(obs)
