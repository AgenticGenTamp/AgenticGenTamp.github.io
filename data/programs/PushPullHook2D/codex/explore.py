from env_client import make_env
import numpy as np

env = make_env()
print("max_steps", env.max_steps)
for seed in range(20):
    o, info = env.reset(seed=seed)
    print(seed, "rob", np.round(o[[0,1,2,3,4,5,7,8]],3),
          "hook", np.round(o[9:20],3), "mov", np.round(o[[20,21,28]],3),
          "tgt", np.round(o[[29,30,37]],3), "info", info)
env.close()
