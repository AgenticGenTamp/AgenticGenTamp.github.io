import numpy as np
from env_client import make_env


def compact(o):
    cubes = np.asarray([[o[16*i], o[16*i+1], o[16*i+2]] for i in range(5)])
    return {
        "cubes": np.round(cubes, 3).tolist(),
        "cook": np.round(o[80:83], 3).tolist(),
        "island": np.round(o[96:99], 3).tolist(),
        "drawers": np.round(o[103:109], 3).tolist(),
        "base": np.round(o[125:128], 3).tolist(),
        "joints": np.round(o[128:136], 3).tolist(),
        "wiper": np.round(o[147:163], 3).tolist(),
    }


for seed in range(8):
    env = make_env()
    obs, info = env.reset(seed=seed)
    print(seed, compact(obs), "info", info)
    env.close()
