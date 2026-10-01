import sys
import numpy as np
from env_client import make_env


def v(s, n, f):
    return float(s.get(s.get_object_from_name(n), f))


def xy(s, n):
    return np.array([v(s, n, "x"), v(s, n, "y")])


count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
env = make_env()
s, info = env.reset(seed=seed, options={"object_count": count})
names = sorted(n for n in s.get_object_names() if n.startswith("cube_"))
p0 = {n: xy(s, n) for n in ["wiper_0"] + names}
print("initial", {n: p.round(3).tolist() for n, p in p0.items()})
for k in range(250):
    rob = np.array([v(s, "robot", "pos_base_x"), v(s, "robot", "pos_base_y")])
    w = xy(s, "wiper_0")
    # Stay about 20 cm north of the wiper and follow it toward negative y.
    goal = w + np.array([0.0, 0.20])
    delta = goal - rob
    a = np.zeros(11, np.float32)
    a[:2] = np.clip(0.8 * delta + np.array([0.0, -0.035]), -0.06, 0.06)
    s, r, done, trunc, inf = env.step(a)
    if k % 10 == 9 or r != -1.0 or done:
        moved = {n: (xy(s, n) - p0[n]).round(3).tolist()
                 for n in p0 if np.linalg.norm(xy(s, n) - p0[n]) > .002}
        print(k + 1, "r", r, "rob", rob.round(2).tolist(), "w", w.round(2).tolist(),
              "moved", moved, "done", done)
    if done or trunc:
        break
env.close()
