"""Closed-form calibrated Kinova planar inverse kinematics."""
import math
import numpy as np

# Exact planar link sums with joint_1=joint_5=0, joint_3=-pi.
_L1 = .42076
_L2 = .31436
_BASE_X = .119899783
# Flange/tool fixed z contribution -.002645034, calibrated base z -.005199842.
_Z0 = -.007844876

def vertical_ik(z, x=.5):
    """Return the elbow-negative, vertical-tool 7-joint configuration.

    x and z are calibrated EE coordinates relative to the mobile base's
    observed x/y translation. Out-of-reach points project to the reachable
    annulus while preserving the vertical orientation.
    """
    px = float(x) - _BASE_X
    pz = float(z) - _Z0
    c = (px * px + pz * pz - _L1 * _L1 - _L2 * _L2) / (2 * _L1 * _L2)
    q4 = -math.acos(max(-1., min(1., c)))
    q2 = math.atan2(px, pz) + math.atan2(_L2 * math.sin(q4), _L1 + _L2 * math.cos(q4))
    q6 = q2 - q4 - math.pi
    return np.array([0., q2, -math.pi, q4, 0., q6, math.pi / 2])

if __name__ == '__main__':
    from kinematics import fk
    import time
    rng = np.random.default_rng(41)
    max_error = 0.
    for _ in range(1000):
        q2 = rng.uniform(-1., 1.5)
        q4 = rng.uniform(-2.6, -.1)
        q = [0., q2, -math.pi, q4, 0., q2-q4-math.pi, math.pi/2]
        t = fk(q)
        p = t[:3, 3] + t[:3, :3] @ [0., 0., -.181525034] + [.119899783, .000348925, -.005199842]
        solved = vertical_ik(p[2], p[0])
        t2 = fk(solved)
        p2 = t2[:3, 3] + t2[:3, :3] @ [0., 0., -.181525034] + [.119899783, .000348925, -.005199842]
        max_error = max(max_error, float(np.max(abs(p-p2))))
        assert np.max(abs(t2[:3, 2]-[0.,0.,1.])) < 1e-10
    assert max_error < 1e-10, max_error
    started = time.perf_counter()
    for _ in range(10000):
        vertical_ik(.16, .5)
    print('1000 reachable points: max Cartesian error', max_error)
    print('Mean call microseconds:', (time.perf_counter()-started)*100)
