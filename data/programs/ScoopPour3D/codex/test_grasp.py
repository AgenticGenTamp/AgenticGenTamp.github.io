"""Test whether the pre-positioned gripper captures the scoop."""

import sys
import numpy as np
from env_client import make_env


FS = ("x", "y", "z", "qw", "qx", "qy", "qz")


def ov(s, name):
    o = s.get_object_from_name(name)
    return np.array([s.get(o, f) for f in FS])


def main(dim, sign, steps=40):
    env = make_env()
    s, info = env.reset(seed=5)
    p0 = ov(s, "scoop_0")
    total = 0.0
    for t in range(steps):
        a = np.zeros(11, np.float32)
        # The scoop starts grasped: zero is the closed command.
        a[10] = 0.0
        if t >= 5:
            a[dim] = sign * 0.1
        s, r, done, trunc, _ = env.step(a)
        total += r
    print("dim", dim, sign, "scoop delta", np.round(ov(s, "scoop_0")-p0, 4), "reward", total)
    env.close()


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 40)
