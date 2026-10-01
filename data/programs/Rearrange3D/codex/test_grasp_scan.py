"""Align home tool over an object and sweep shoulder for empirical contact."""
import numpy as np
from env_client import make_env

env = make_env()
for which, ix in (("box", 16), ("can", 32)):
    for direction in (-1.0, 1.0):
        s, _ = env.reset(seed=0)
        start = s.copy()
        # Home FK tool offset is approximately (+.178, 0) from reported base.
        bx, by = float(s[ix] - .178), float(s[ix + 1])
        for k in range(10):
            a = np.zeros(11, np.float32)
            a[0] = np.clip((bx-s[93]) * .12, -.1, .1)
            a[1] = np.clip((by-s[94]) * .12, -.1, .1)
            s, r, t, tr, info = env.step(a)
        aligned = s.copy()
        print(which, direction, "base", np.round(s[93:96],3).tolist())
        # Sweep shoulder joint (action arm index 1 => action 4), open.
        for k in range(35):
            a = np.zeros(11, np.float32); a[4] = direction*.1
            s, r, t, tr, info = env.step(a)
            d = np.max(np.abs(s[[0,1,2,16,17,18,32,33,34]]-start[[0,1,2,16,17,18,32,33,34]]))
            if d > .002 or k in (9,19,29,34):
                print(" k", k, "q", round(float(s[97]),3), "r",r,"maxobj",round(float(d),4))
                if d > .002: break
env.close()
