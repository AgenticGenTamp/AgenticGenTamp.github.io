import numpy as np


def robot_polys(x, y, th, L, gap):
    """Return dict of corner arrays (4x2) for gripper base, wall(+v) finger, other(-v) finger."""
    u = np.array([np.cos(th), np.sin(th)]); v = np.array([-np.sin(th), np.cos(th)])
    G = np.array([x, y]) + L * u
    def rect(c, hu, hv):
        return np.array([c + su * hu * u + sv * hv * v for su in (-1, 1) for sv in (-1, 1)])
    return {"plate": rect(G, 0.03, 0.16),
            "fp": rect(G + 0.13 * u + gap / 2 * v, 0.1, 0.03),
            "fm": rect(G + 0.13 * u - gap / 2 * v, 0.1, 0.03),
            "arm": rect(G - (0.03 + 0.24) * u, 0.24, 0.03)}
