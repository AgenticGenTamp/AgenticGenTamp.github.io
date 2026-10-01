"""One-connection action response probes (reset between each trial)."""

import numpy as np
from env_client import make_env


env = make_env()


def trial(seed, acts):
    s0, _ = env.reset(seed=seed)
    s = s0
    out = []
    for a in acts:
        s, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        out.append((r, term, trunc))
    return s0, s, out


base = np.zeros(11, np.float32)
for i in range(11):
    a = base.copy()
    a[i] = 0.1 if i < 10 else 1.0
    s0, s1, out = trial(0, [a])
    print(i, "r", out, "robot delta", np.round(s1[93:104]-s0[93:104], 4).tolist(),
          "object delta", np.round(np.r_[s1[:3]-s0[:3], s1[16:19]-s0[16:19], s1[32:35]-s0[32:35]], 4).tolist())

for i in range(11):
    a = base.copy()
    a[i] = -0.1 if i < 10 else 1.0
    s0, s1, out = trial(0, [a] * 5)
    print("repeat", i, "r", out[-1], "robot delta", np.round(s1[93:104]-s0[93:104], 4).tolist())

env.close()
