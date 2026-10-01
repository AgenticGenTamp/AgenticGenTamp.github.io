"""Nominal Kinova Gen3 chain; tool/mount transforms need empirical calibration.
Based on remembered public Gen3 URDF geometry, not environment internals.
"""
import numpy as np
from scipy.spatial.transform import Rotation

JOINT_XYZ = np.array([
    [0, 0, .15643],
    [0, .005375, -.12838],
    [0, -.21038, -.006375],
    [0, .006375, -.21038],
    [0, -.20843, -.006375],
    [0, 0, -.10593],
    [0, -.10593, 0],
])
JOINT_RX = np.array([np.pi, np.pi/2, -np.pi/2, np.pi/2,
                     -np.pi/2, np.pi/2, -np.pi/2])

def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    out = np.eye(4)
    out[:3,:3] = Rotation.from_euler('xyz', rpy).as_matrix()
    out[:3,3] = xyz
    return out

JOINT_FIXED = [transform(p, (rx, 0, 0)) for p,rx in zip(JOINT_XYZ, JOINT_RX)]


def forward(q, finger_extension=.12, mount_xyz=(0, 0, 0)):
    """World-from-tool; finger_extension measured from flange along tool +Z.

    The Gen3 base is frame zero. Add robot's arm mount height/offset separately.
    Positive tool Z points out of the flange toward the fingers.
    """
    t = transform(mount_xyz)
    for fixed, qi in zip(JOINT_FIXED, q):
        t = t @ fixed @ transform(rpy=(0, 0, qi))
    t = t @ transform((0, 0, -.061525), (np.pi, 0, 0))
    return t @ transform((0, 0, finger_extension))

if __name__ == '__main__':
    q = np.array([0, -.3491, np.pi, -2.5482, 0, -.8727, np.pi/2])
    print('flange', forward(q, 0))
    print('nominal fingers', forward(q)[:3,3])
