import math
import numpy as np
from env_client import make_env

env = make_env()
s, _ = env.reset(seed=1)
for i in range(30):
    r = s.get_object_from_name("robot")
    b = s.get_object_from_name("block1")
    rx, ry = (float(s.get(r, q)) for q in ("x", "y"))
    bx, by = (float(s.get(b, q)) for q in ("x", "y"))
    th, arm = (float(s.get(r, q)) for q in ("theta", "arm_joint"))
    goal = math.atan2(by-ry, bx-rx)
    err = (goal-th+math.pi)%(2*math.pi)-math.pi
    dist = math.hypot(bx-rx, by-ry)
    vac = 0 if i < 5 else 1
    da = 0.0 if i >= 5 else max(-.1, min(.1, dist-arm))
    dx = .03 if i >= 7 else 0.0
    a = np.array([dx, 0, max(-.196, min(.196, err)) if i < 7 else 0, da, vac], np.float32)
    s, rew, done, trunc, _ = env.step(a)
    print(i, "act", a.round(3), "r", [round(float(s.get(r,q)),3) for q in ("x","y","theta","arm_joint","vacuum")], "b", [round(float(s.get(b,q)),3) for q in ("x","y","theta")])
env.close()
