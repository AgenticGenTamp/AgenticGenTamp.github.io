"""Probe base target coordinates, wrapping, and collision boundaries."""

import numpy as np

from env_client import make_env


def xyz(s):
    r = s.get_object_from_name("robot")
    return np.array([s.get(r, f) for f in ("base_x", "base_y", "base_rot")])


def main():
    env = make_env()
    try:
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            s, _ = env.reset(seed=0)
            out = [xyz(s)]
            for _ in range(20):
                a = np.zeros(11, dtype=np.float32)
                a[axis] = 0.2 * sign
                s, *_ = env.step(a)
                out.append(xyz(s))
            print("walk", axis, sign, np.round(np.array(out), 3).tolist())

        s, _ = env.reset(seed=0)
        for _ in range(8):
            a = np.zeros(11, dtype=np.float32); a[2] = 0.2
            s, *_ = env.step(a)
        before = xyz(s)
        a = np.zeros(11, dtype=np.float32); a[0] = 0.2
        s, *_ = env.step(a)
        print("after yaw ~pi/2 then +x", np.round(before, 5), "->", np.round(xyz(s), 5))
        a = np.zeros(11, dtype=np.float32); a[1] = 0.2
        s, *_ = env.step(a)
        print("then +y", np.round(xyz(s), 5))

        s, _ = env.reset(seed=0)
        vals = []
        for _ in range(40):
            a = np.zeros(11, dtype=np.float32); a[2] = 0.2
            s, *_ = env.step(a); vals.append(xyz(s)[2])
        print("rotation walk", np.round(vals, 3).tolist())
    finally:
        env.close()


if __name__ == "__main__":
    main()
