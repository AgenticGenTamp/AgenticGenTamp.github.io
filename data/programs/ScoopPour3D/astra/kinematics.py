"""Candidate Kinova Gen3 kinematics from public URDF dimensions.

Mount height and gripper extension are configurable because simulation mounting
and gripper geometry have not yet been empirically calibrated.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

_ORIGINS = [((0, 0, .15643), np.pi),
            ((0, .005375, -.12838), np.pi / 2),
            ((0, -.21038, -.006375), -np.pi / 2),
            ((0, .006375, -.21038), np.pi / 2),
            ((0, -.20843, -.006375), -np.pi / 2),
            ((0, 0, -.10593), np.pi / 2),
            ((0, -.10593, 0), -np.pi / 2)]
_ORIGIN_MATS = []
for xyz, rx in _ORIGINS:
    t = np.eye(4)
    t[:3, 3] = xyz
    t[:3, :3] = Rotation.from_euler('x', rx).as_matrix()
    _ORIGIN_MATS.append(t)

def fk(q, base=(0., 0., 0.), mount_z=.4, extension=.12):
    """Return world transform at gripper point; extension along tool +z."""
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('z', base[2]).as_matrix()
    t[:3, 3] = [base[0] + .12 * np.cos(base[2]), base[1] + .12 * np.sin(base[2]), mount_z]
    for o, v in zip(_ORIGIN_MATS, q):
        r = np.eye(4)
        c, s = np.cos(v), np.sin(v)
        r[:2, :2] = [[c, -s], [s, c]]
        t = t @ o @ r
    o = np.eye(4)
    o[:3, 3] = [0, 0, -.061525 - extension]
    o[:3, :3] = Rotation.from_euler('x', np.pi).as_matrix()
    return t @ o

def ik(target_xyz, target_rotation, q0, base=(0., 0., 0.), mount_z=.4, extension=.12, max_nfev=80):
    target_xyz = np.asarray(target_xyz)
    q0 = np.asarray(q0)
    def error(q):
        t = fk(q, base, mount_z, extension)
        pos = (t[:3, 3] - target_xyz)
        rot = Rotation.from_matrix(target_rotation @ t[:3, :3].T).as_rotvec()
        return np.r_[pos, .15 * rot, .002 * (q - q0)]
    opt = least_squares(error, q0, max_nfev=max_nfev, ftol=1e-5, xtol=1e-5)
    return opt.x, np.linalg.norm(error(opt.x)[:3])
