"""Measure arm-axis effects after lowering into scoop contact."""

import sys
import numpy as np
from env_client import make_env


def pose(s):
    o = s.get_object_from_name("scoop_0")
    return np.array([s.get(o, f) for f in ("x", "y", "z")])


def main(dim, sign, sweep_steps=13, grip=1.0, follow_y=0.0, follow_x=0.0,
         follow_steps=10, follow_yaw=0.0):
    env = make_env()
    s, _ = env.reset(seed=0)
    p0 = pose(s)
    for t in range(32 + sweep_steps + follow_steps):
        a = np.zeros(11, np.float32)
        a[10] = 1.0
        if t < 32:
            a[4], a[6] = .1, .06
        elif t < 32 + sweep_steps:
            a[10] = grip
            a[dim] = sign * .1
        else:
            a[10] = grip
            a[0], a[1], a[2] = follow_x, follow_y, follow_yaw
        s, _, _, _, _ = env.step(a)
    print(dim, sign, sweep_steps, grip, follow_y, follow_x, follow_yaw,
          np.round(pose(s) - p0, 4), np.round(pose(s), 4))
    env.close()


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]),
         int(sys.argv[3]) if len(sys.argv) > 3 else 13,
         float(sys.argv[4]) if len(sys.argv) > 4 else 1.0,
         float(sys.argv[5]) if len(sys.argv) > 5 else 0.0,
         float(sys.argv[6]) if len(sys.argv) > 6 else 0.0,
         int(sys.argv[7]) if len(sys.argv) > 7 else 10,
         float(sys.argv[8]) if len(sys.argv) > 8 else 0.0)
