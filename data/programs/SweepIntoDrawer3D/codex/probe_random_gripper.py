import numpy as np
from env_client import make_env

env = make_env()
o, _ = env.reset(seed=0)
for target in (1.0, 0.0, 0.5):
    vals = []
    for _ in range(12):
        a = np.zeros(11, np.float32)
        a[10] = target
        o, r, term, trunc, _ = env.step(a)
        vals.append(round(float(o[135]), 4))
    print("command", target, "observed", vals)
env.close()
