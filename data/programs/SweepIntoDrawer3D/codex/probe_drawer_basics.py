import numpy as np
from env_client import make_env


def snap(o):
    return {
        "cubes": np.asarray(o).reshape(-1)[np.array([[16*i,16*i+1,16*i+2] for i in range(5)])].round(3).tolist(),
        "island": np.asarray(o)[96:109].round(3).tolist(),
        "robot": np.asarray(o)[125:147].round(3).tolist(),
        "wiper": np.asarray(o)[147:163].round(3).tolist(),
    }


for seed in range(3):
    env = make_env()
    o, info = env.reset(seed=seed)
    print("seed", seed, "max", env.max_steps, "initial", snap(o), "info", info)
    for label, action in [
        ("zero", np.zeros(11, np.float32)),
        ("base+x", np.array([.1,0,0,0,0,0,0,0,0,0,0], np.float32)),
        ("j1+", np.array([0,0,0,.1,0,0,0,0,0,0,0], np.float32)),
    ]:
        before = o.copy()
        o, r, term, trunc, inf = env.step(action)
        print(label, "r", r, "diff robot", (o[125:147]-before[125:147]).round(4).tolist(), "wiper", o[147:150].round(3).tolist(), "drawers",o[103:109].round(3).tolist())
    env.close()
