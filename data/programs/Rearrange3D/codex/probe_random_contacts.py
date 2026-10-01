"""Random coordinated arm motions after driving up to the counter."""

import numpy as np
from env_client import make_env


OBJ = np.array([0, 1, 2, 16, 17, 18, 32, 33, 34])
env = make_env()
rng = np.random.default_rng(9182)
for trial in range(4):
    s0, _ = env.reset(seed=0)
    a = np.zeros(11, np.float32)
    # Vary lateral alignment, then approach until counter collision.
    a[1] = (-0.1, -0.05, 0.05, 0.1)[trial]
    for _ in range((0, 3, 3, 5)[trial]):
        s, *_ = env.step(a)
    a[:] = 0
    a[0] = 0.1
    for _ in range(6):
        s, *_ = env.step(a)
    ref = s[OBJ].copy()
    print("trial", trial, "base", np.round(s[93:96], 3).tolist())
    # Piecewise-constant commands produce substantial coordinated sweeps.
    for block in range(12):
        a[:] = 0
        a[3:10] = rng.choice([-0.1, 0.0, 0.1], size=7)
        a[10] = float(block % 2)
        for k in range(5):
            s, r, term, trunc, info = env.step(a)
            d = (s[OBJ] - ref).reshape(3, 3)
            if np.max(np.abs(d)) > 0.004:
                print("CONTACT", trial, block, k, "r", r,
                      "cmd", a.tolist(), "q", np.round(s[96:103], 2).tolist(),
                      "delta", np.round(d, 4).tolist())
                break
env.close()
