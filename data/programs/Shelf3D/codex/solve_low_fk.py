"""Offline IK candidates compensating the observed base-height FK offset."""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation

ORIGINS = [
    ([0, 0, .15643], [np.pi, 0, 0]),
    ([0, .005375, -.12838], [np.pi / 2, 0, np.pi]),
    ([0, -.21038, -.006375], [-np.pi / 2, 0, np.pi]),
    ([0, .006375, -.21038], [np.pi / 2, 0, np.pi]),
    ([0, -.20843, -.006375], [-np.pi / 2, 0, np.pi]),
    ([0, 0, .10593], [np.pi / 2, 0, np.pi]),
    ([0, -.10593, 0], [-np.pi / 2, 0, np.pi]),
]


def tr(x, r):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", r).as_matrix()
    out[:3, 3] = x
    return out


def fk(q):
    out = np.eye(4)
    for qi, (x, r) in zip(q, ORIGINS):
        out = out @ tr(x, r) @ tr([0, 0, 0], [0, 0, qi])
    return out @ tr([0, 0, -.0615], [0, 0, 0])


bounds = [(-np.pi, np.pi), (-2.24, 2.24), (-np.pi, np.pi),
          (-2.58, 2.58), (-np.pi, np.pi), (-2.1, 2.1),
          (-np.pi, np.pi)]
for z in (-.35, -.45, -.55):
    def cost(q):
        t = fk(q)
        p, axis = t[:3, 3], t[:3, 2]
        return ((p[0] - .30) ** 2 + p[1] ** 2 + (p[2] - z) ** 2
                + .001 * np.sum((axis - [0, 0, -1]) ** 2))

    result = differential_evolution(cost, bounds, seed=4, popsize=10,
                                    maxiter=180, tol=1e-7)
    t = fk(result.x)
    print("z", z, "cost", round(result.fun, 5), "q", np.round(result.x, 3),
          "p", np.round(t[:3, 3], 3), "axis", np.round(t[:3, 2], 3))
