"""Estimate reachable wrapped joint extrema with simultaneous commands."""

import numpy as np
from env_client import make_env


env = make_env()
for sign in (-1.0, 1.0):
    s, _ = env.reset(seed=0)
    a = np.zeros(11, np.float32)
    a[3:10] = sign * .1
    last = s[96:103].copy()
    for k in range(100):
        s, r, term, trunc, info = env.step(a)
        if k in (9, 29, 59, 99):
            print(sign, k + 1, np.round(s[96:103], 3).tolist(),
                  "stepdelta", np.round(s[96:103] - last, 3).tolist())
        last = s[96:103].copy()
env.close()
