"""Try using the lowered wrist directly as a source-to-target paddle."""

import sys
import numpy as np
from env_client import make_env


def get(s, name, fs):
    o = s.get_object_from_name(name)
    return np.array([s.get(o, f) for f in fs])


def main(seed):
    env = make_env(); s, _ = env.reset(seed=seed)
    for t in range(100):
        a = np.zeros(11, np.float32); a[10] = 1
        if t < 25:
            p = get(s, "robot", ("pos_base_x", "pos_base_y", "pos_base_rot"))
            a[:3] = np.clip(.5 * (np.array([-.06, -.20, 0.0]) - p), -.1, .1)
        elif t < 60:
            a[4], a[6] = .1, .06
        else:
            a[1] = .1
        s, r, done, trunc, _ = env.step(a)
    cs = np.array([get(s, n, ("x", "y", "z")) for n in s.get_object_names() if n.startswith("cube_")])
    print(seed, r, np.round(cs.min(0), 3), np.round(cs.mean(0), 3), np.round(cs.max(0), 3))
    env.close()


if __name__ == "__main__": main(int(sys.argv[1]))
