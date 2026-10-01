"""Search simple single-axis motions for contacts with task objects."""

import numpy as np
from env_client import make_env


env = make_env()
obj_idx = np.array([0, 1, 2, 16, 17, 18, 32, 33, 34])
for grip in (0.0, 1.0):
    for i in range(10):
        for sign in (-1.0, 1.0):
            s0, _ = env.reset(seed=0)
            a = np.zeros(11, np.float32)
            a[i] = sign * 0.1
            a[10] = grip
            s = s0
            reward = None
            for _ in range(6):
                s, reward, term, trunc, info = env.step(a)
            d = s[obj_idx] - s0[obj_idx]
            mag = np.max(np.abs(d.reshape(3, 3)), axis=1)
            if np.max(mag) > 0.002 or reward != -1.0:
                print("grip", grip, "axis", i, "sign", sign, "reward", reward,
                      "object maxdelta", np.round(mag, 4).tolist(),
                      "xyzdelta", np.round(d, 4).tolist())
env.close()
