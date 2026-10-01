"""Search simple descend-close-retract scoop pickup timings."""

import sys
import numpy as np
from env_client import make_env


def pose(s, name):
    o = s.get_object_from_name(name)
    return np.array([s.get(o, f) for f in ("x", "y", "z", "qw", "qx", "qy", "qz")])


def main(n, elbow=0.0, wrist_steps=0, base_y=0.0, base_x=0.0):
    env = make_env()
    s, _ = env.reset(seed=0)
    p0 = pose(s, "scoop_0")
    # Close during the final two descent increments, before an open fingertip
    # can shove the diagonal handle away.
    phases = [(abs(wrist_steps), 1.0, 0.0, np.sign(wrist_steps) * .1),
              (3, 1.0, 0.0, 0.0), (max(0, n - 2), 1.0, .1, 0.0),
              (2, 0.0, .1, 0.0), (6, 0.0, 0.0, 0.0),
              (15, 0.0, -.1, 0.0)]
    if base_y or base_x:
        a = np.zeros(11, np.float32)
        a[0], a[1], a[10] = base_x, base_y, 1.0
        s, _, _, _, _ = env.step(a)
    for count, grip, shoulder, wrist in phases:
        for _ in range(count):
            a = np.zeros(11, np.float32)
            a[10] = grip
            a[4] = shoulder
            if shoulder:
                a[6] = np.sign(shoulder) * elbow
            a[9] = wrist
            s, r, done, trunc, _ = env.step(a)
    print(n, elbow, wrist_steps, base_y, base_x, "delta", np.round(pose(s, "scoop_0") - p0, 4), "final", np.round(pose(s, "scoop_0")[:3], 4))
    env.close()


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 0.0,
         int(sys.argv[3]) if len(sys.argv) > 3 else 0,
         float(sys.argv[4]) if len(sys.argv) > 4 else 0.0,
         float(sys.argv[5]) if len(sys.argv) > 5 else 0.0)
