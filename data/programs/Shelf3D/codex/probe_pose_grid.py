"""Coarse planar arm-pose search on the deterministic one-cube layout."""

import itertools
import numpy as np
from env_client import make_env

JF = [f"pos_arm_joint{i}" for i in range(1, 8)]


def vals(s):
    r = s.get_object_from_name("robot")
    c = s.get_object_from_name("cube1")
    q = np.array([s.get(r, f) for f in JF], float)
    p = np.array([s.get(c, f) for f in ("x", "y", "z")], float)
    return q, p


def servo(e, s, target, grip, steps=55):
    for _ in range(steps):
        q, _ = vals(s)
        a = np.zeros(11, np.float32)
        a[3:10] = np.clip(1.8 * (target - q), -.1, .1)
        a[10] = grip
        s, rew, term, trunc, _ = e.step(a)
    return s


home = np.array([0, -.3491, np.pi, -2.5482, 0, -.8727, np.pi / 2])
for q2, q4, q6 in itertools.product((-.9, -.4, .1, .6), (-2.2, -1.5), (-1.5, -.7)):
    e = make_env()
    s, _ = e.reset(seed=0, options={"object_count": 1})
    _, p0 = vals(s)
    target = home.copy(); target[[1, 3, 5]] = (q2, q4, q6)
    s = servo(e, s, target, 0.0)
    q, p_reach = vals(s)
    s = servo(e, s, target, 1.0, 12)
    s = servo(e, s, home, 1.0)
    _, p_lift = vals(s)
    print("pose", q2, q4, q6, "reach", np.round(p_reach-p0,3),
          "lift", np.round(p_lift-p0,3), "final", np.round(p_lift,3))
    e.close()
