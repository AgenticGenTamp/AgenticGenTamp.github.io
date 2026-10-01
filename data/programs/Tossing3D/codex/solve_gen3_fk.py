"""Small independent Gen3 URDF FK/IK calculator (offline only)."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ORIGINS = (
    ((0, 0, .15643), (np.pi, 0, 0)),
    ((0, .005375, -.12838), (np.pi / 2, 0, np.pi)),
    ((0, -.21038, -.006375), (-np.pi / 2, 0, np.pi)),
    ((0, .006375, -.21038), (np.pi / 2, 0, np.pi)),
    ((0, -.20843, -.006375), (-np.pi / 2, 0, np.pi)),
    ((0, 0, .10593), (np.pi / 2, 0, np.pi)),
    ((0, -.10593, 0), (-np.pi / 2, 0, np.pi)),
)


def transform(xyz, rpy):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    t[:3, 3] = xyz
    return t


def fk(q):
    t = np.eye(4)
    for origin, angle in zip(ORIGINS, q):
        t = t @ transform(*origin)
        rz = np.eye(4)
        rz[:3, :3] = Rotation.from_rotvec([0, 0, angle]).as_matrix()
        t = t @ rz
    # Robotiq/Kinova hand attachment estimate.
    t = t @ transform((0, 0, -.0615), (0, 0, 0))
    return t


def residual(q, position):
    t = fk(q)
    # Tool Z axis vertical downward. Keep yaw unconstrained.
    zaxis = t[:3, 2]
    return np.r_[12 * (t[:3, 3] - position), zaxis[:2], zaxis[2] + 1]


HOME = np.array([0., -.3491, np.pi, -2.5482, 0., -.8727, np.pi / 2])
OLD = np.array([.23, .39, 3.40, 0., .17, -1.57, 2.46])
print("home", np.round(fk(HOME)[:3, 3], 4), "zaxis", np.round(fk(HOME)[:3, 2], 3))
print("old ", np.round(fk(OLD)[:3, 3], 4), "zaxis", np.round(fk(OLD)[:3, 2], 3))
for z in (-.15, -.25, -.35):
    best = None
    for seed in range(20):
        q0 = np.random.RandomState(seed).uniform(-np.pi, np.pi, 7)
        result = least_squares(residual, q0, args=(np.array([.45, 0., z]),),
                               bounds=(-2 * np.pi, 2 * np.pi), max_nfev=1000)
        score = np.linalg.norm(residual(result.x, np.array([.45, 0., z])))
        if best is None or score < best[0]:
            best = score, result.x, fk(result.x)
    print("target z", z, "score", best[0], "q", np.round(best[1], 4),
          "xyz", np.round(best[2][:3, 3], 4), "zaxis", np.round(best[2][:3, 2], 3))
