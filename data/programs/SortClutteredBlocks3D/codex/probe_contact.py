"""Locate the first contact along the successful arm/base sweep."""
from env_client import make_env
import numpy as np

env = make_env(); s, _ = env.reset(seed=0, options={"object_count": 4})
r = s.get_object_from_name("robot")
names = sorted(n for n in s.get_object_names() if n.startswith("cube"))
initial = {n: np.array([s.get(s.get_object_from_name(n), f)
                        for f in ("x", "y", "z")]) for n in names}
qgoal = np.array([0., 1.3, np.pi, -1.7, 0., 1., np.pi / 2])

for t in range(160):
    robot = s.get_object_from_name("robot")
    b = np.array([s.get(robot, f) for f in
                  ("pos_base_x", "pos_base_y", "pos_base_rot")])
    q = np.array([s.get(robot, "pos_arm_joint%d" % i) for i in range(1, 8)])
    goal = np.array([1., initial["cube3"][1], np.pi]) if t < 100 else np.array([.48, initial["cube3"][1], np.pi])
    a = np.zeros(11, np.float32)
    a[:3] = np.clip(1.5 * (goal - b), -.1, .1)
    a[3:10] = np.clip(1.5 * (qgoal - q), -.1, .1)
    a[10] = 0.
    s, rew, _, _, _ = env.step(a)
    changes = []
    for n in names:
        o = s.get_object_from_name(n)
        p = np.array([s.get(o, f) for f in ("x", "y", "z")])
        if np.linalg.norm(p - initial[n]) > .003:
            changes.append((n, np.round(p, 3).tolist()))
    if changes:
        print(t + 1, np.round(b, 3), rew, changes)
env.close()
