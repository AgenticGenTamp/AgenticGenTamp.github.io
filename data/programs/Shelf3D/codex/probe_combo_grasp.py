"""Test visually combined low-pose features over a coarse base grid."""

import numpy as np
from env_client import make_env
from probe_grasp_sequence import drive, pose, value

LOW = np.array([0.0, 1.9, 0.0, -2.55, 2.0, -2.1, np.pi / 2])
HOME = np.array([0.0, -.3491, np.pi, -2.5482, 0.0, -.8727, np.pi / 2])

for dx in (0.0, .12, .24, .36):
    for dy in (-.12, 0.0, .12):
        e = make_env(); s, _ = e.reset(seed=0, options={"object_count": 1})
        p0 = pose(s, "cube1")
        for _ in range(8):
            a = np.zeros(11, np.float32); a[0] = dx / 8 / .87; a[1] = dy / 8 / .87
            s, *_ = e.step(a)
        s = drive(e, s, LOW, 0.0, 100)
        p1 = pose(s, "cube1")
        for _ in range(8):
            a = np.zeros(11, np.float32); a[10] = 1.; s, *_ = e.step(a)
        s = drive(e, s, HOME, 1.0, 100)
        p2 = pose(s, "cube1")
        print(dx, dy, np.round(p1-p0, 3), np.round(p2-p0, 3))
        e.close()
