"""Probe each control after the policy's calibrated source-approach phase."""

import sys
import numpy as np
from approach import GeneratedApproach
from env_client import make_env


def xyz(s, name):
    o = s.get_object_from_name(name)
    return np.array([s.get(o, f) for f in ("x", "y", "z")])


def main(dim, sign, steps=20, dim2=-1, sign2=0.0):
    env = make_env(); s, info = env.reset(seed=0)
    p = GeneratedApproach(env.action_space, env.observation_space, {})
    p.reset(s, info)
    for _ in range(67):
        s, _, _, _, _ = env.step(p.get_action(s))
    # Reproduce the contact-producing source feed from approach af78e0e.
    for _ in range(12):
        a = np.zeros(11, np.float32); a[10] = 1.0
        a[:2] = np.clip(.035 * p.source_dir, -.05, .05)
        a[2] = -.1 if p.source_dir[1] < 0 else .1
        s, _, _, _, _ = env.step(a)
    y0 = xyz(s, "bin_yellow_0"); g0 = xyz(s, "bin_green_0")
    best = -99.0
    for _ in range(steps):
        a = np.zeros(11, np.float32); a[10] = 1.0; a[dim] = sign * .1
        if dim2 >= 0:
            a[dim2] = sign2 * .1
        s, r, done, trunc, _ = env.step(a); best = max(best, r)
    print(dim, sign, dim2, sign2, "yellow", np.round(xyz(s, "bin_yellow_0")-y0, 4),
          "green", np.round(xyz(s, "bin_green_0")-g0, 4),
          "reward", best, "done", done)
    env.close()


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv)>3 else 20,
         int(sys.argv[4]) if len(sys.argv)>4 else -1,
         float(sys.argv[5]) if len(sys.argv)>5 else 0.0)
