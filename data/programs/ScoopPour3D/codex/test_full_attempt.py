"""Exercise a complete paddle-to-source then source-to-target sweep."""

import sys
import numpy as np
from env_client import make_env


def xyz(s, name):
    o = s.get_object_from_name(name)
    return np.array([s.get(o, f) for f in ("x", "y", "z")])


def main(seed=0):
    env = make_env(); s, info = env.reset(seed=seed)
    phases = [
        (32, {4: .1, 6: .06}),
        (5, {8: .1}),
        (10, {1: -.03, 2: -.1}),
        (15, {0: .06, 2: .1}),
        (15, {5: .1, 6: .1}),
        (120, {1: .1}),
    ]
    total = 0; t = 0
    for count, cmds in phases:
        for _ in range(count):
            a = np.zeros(11, np.float32); a[10] = 1
            for d, v in cmds.items(): a[d] = v
            s, r, done, trunc, _ = env.step(a); total += r; t += 1
        cubes = np.array([xyz(s, n) for n in s.get_object_names() if n.startswith("cube_")])
        print(t, "r", r, "scoop", np.round(xyz(s, "scoop_0"), 3),
              "cubes", np.round(cubes.min(0), 3), np.round(cubes.mean(0), 3), np.round(cubes.max(0), 3))
    print("total", total)
    env.close()


if __name__ == "__main__": main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
