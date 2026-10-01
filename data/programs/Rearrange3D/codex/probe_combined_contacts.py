"""Combined base approach and broad arm sweeps; stop on object contact."""

import numpy as np
from env_client import make_env


OBJ = np.array([0, 1, 2, 16, 17, 18, 32, 33, 34])
env = make_env()


def step(a, s0, label, k):
    s, r, term, trunc, info = env.step(a)
    d = (s[OBJ] - s0[OBJ]).reshape(3, 3)
    if np.max(np.abs(d)) > 0.003 or r != -1.0:
        print("CONTACT", label, k, "reward", r, "base", np.round(s[93:96], 3).tolist(),
              "q", np.round(s[96:103], 2).tolist(), "obj delta", np.round(d, 4).tolist())
    return s


for base_steps in (0, 6):
    for arm_axis in range(6, 10):
        s0, _ = env.reset(seed=0)
        s = s0
        a = np.zeros(11, np.float32)
        a[0] = 0.1
        for k in range(base_steps):
            s = step(a, s0, "base", k)
        if arm_axis == 3:
            print("approach", base_steps, "base achieved", np.round(s[93:96], 3).tolist())
        # Open during approach, close while sweeping in case an object enters fingers.
        a[:] = 0
        a[arm_axis] = 0.1
        a[10] = 1
        for k in range(10):
            s = step(a, s0, "b%d q%d+" % (base_steps, arm_axis - 3), k)
        a[arm_axis] = -0.1
        for k in range(20):
            s = step(a, s0, "b%d q%d-" % (base_steps, arm_axis - 3), k)
env.close()
