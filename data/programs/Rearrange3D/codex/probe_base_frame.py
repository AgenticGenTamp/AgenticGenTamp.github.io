"""Determine whether planar base commands are world- or body-frame."""

import numpy as np
from env_client import make_env


env = make_env()
for move_axis in (0, 1):
    s0, _ = env.reset(seed=0)
    a = np.zeros(11, np.float32)
    a[2] = 0.1
    for _ in range(15):
        s, *_ = env.step(a)
    before = s.copy()
    a[:] = 0
    a[move_axis] = 0.1
    for _ in range(3):
        s, *_ = env.step(a)
    print("axis", move_axis, "yaw", round(float(before[95]), 3),
          "world delta", np.round(s[93:95] - before[93:95], 4).tolist(),
          "final", np.round(s[93:96], 3).tolist())
env.close()
