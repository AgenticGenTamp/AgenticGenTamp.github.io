from env_client import make_env
import numpy as np

env = make_env()
o, _ = env.reset(seed=0)
a = np.zeros(11, np.float32)
changes = []
for k in range(1001):
    o, r, t, tr, info = env.step(a)
    if k < 3 or t or tr or r != -1:
        changes.append((k+1, r, t, tr, info, o[:3].tolist(), o[16:19].tolist(), o[32:35].tolist()))
    if t or tr:
        break
print(changes)
env.close()
