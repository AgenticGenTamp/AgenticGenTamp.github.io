"""Candidate standard Kinova Gen3 FK, reconstructed from public model memory.

Transforms are URDF origin followed by rotation about local z. The returned
pose is the tool flange relative to the arm base. Mobile-base mounting and
gripper fingertip offset must be supplied/calibrated by the caller.
"""
import math
import numpy as np

_ORIGINS = (
    ((0., 0., .15643), math.pi),
    ((0., .005375, -.12838), math.pi / 2),
    ((0., -.21038, -.006375), -math.pi / 2),
    ((0., .006375, -.21038), math.pi / 2),
    ((0., -.20843, -.006375), -math.pi / 2),
    ((0., .000175, -.10593), math.pi / 2),
    ((0., -.10593, -.000175), -math.pi / 2),
)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((1., 0., 0.), (0., c, -s), (0., s, c)))


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))


def fk(q, tool_length=0., mount_translation=(0., 0., 0.), mount_yaw=0.):
    """Return (position, orientation), optionally extending along flange +z."""
    p = np.array(mount_translation, dtype=float)
    r = _rz(mount_yaw)
    for angle, (xyz, roll) in zip(q, _ORIGINS):
        p = p + r @ np.array(xyz)
        r = r @ _rx(roll) @ _rz(float(angle))
    p = p + r @ np.array((0., 0., -.061525))
    r = r @ _rx(math.pi)
    p = p + r @ np.array((0., 0., tool_length))
    return p, r


def joint_points(q):
    """Return joint origins in arm coordinates for visual model comparison."""
    p = np.zeros(3)
    r = np.eye(3)
    points = [p.copy()]
    for angle, (xyz, roll) in zip(q, _ORIGINS):
        p = p + r @ np.array(xyz)
        points.append(p.copy())
        r = r @ _rx(roll) @ _rz(float(angle))
    p = p + r @ np.array((0., 0., -.061525))
    points.append(p.copy())
    return np.array(points)
