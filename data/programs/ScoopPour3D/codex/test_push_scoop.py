"""Test using the low end effector to push the scoop without grasping it."""

import sys
import numpy as np
from env_client import make_env


def xyz(s, name):
    o = s.get_object_from_name(name)
    return np.array([s.get(o, f) for f in ("x", "y", "z")])


def main(ax, ay, close_delay=99):
    env = make_env()
    s, _ = env.reset(seed=0)
    p0 = xyz(s, "scoop_0")
    for t in range(75):
        a = np.zeros(11, np.float32)
        a[10] = 1.0
        if t < 33:
            a[4], a[6] = .1, .06
        elif t < 45:
            a[0], a[1] = ax, ay
            if t >= 33 + close_delay:
                a[10] = 0.0
        elif t < 65:
            a[10] = 0.0
            a[4], a[6] = -.1, -.06
        else:
            a[10] = 0.0
            a[1] = .05
        s, r, done, trunc, _ = env.step(a)
        if t >= 28:
            print(t, np.round(xyz(s, "scoop_0"), 3), "base", np.round(xyz(s, "bin_yellow_0"), 3), "r", r)
    print("final", ax, ay, close_delay, np.round(xyz(s, "scoop_0") - p0, 4))
    env.close()


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 99)
