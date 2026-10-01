import numpy as np

def _qmat(w, x, y, z):
    n = np.sqrt(w*w + x*x + y*y + z*z); w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

# Kinova Gen3 chain (mujoco menagerie): (pos, quat) of each joint body; joints rotate about local z
CHAIN = [((0, 0, 0.15643), (0, 1, 0, 0)),
         ((0, 0.005375, -0.12838), (1, 1, 0, 0)),
         ((0, -0.21038, -0.006375), (1, -1, 0, 0)),
         ((0, 0.006375, -0.21038), (1, 1, 0, 0)),
         ((0, -0.20843, -0.006375), (1, -1, 0, 0)),
         ((0, 0.00017505, -0.10593), (1, 1, 0, 0)),
         ((0, -0.10593, -0.00017505), (1, -1, 0, 0))]
CHAIN_R = [(np.array(p, float), _qmat(*q)) for p, q in CHAIN]

def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def fk_arm(q, tool=0.0):
    """Returns (pos, R, joint_positions, joint_axes) in arm base frame. Tool point at local -z * (0.0615+tool)? use tool offset along bracelet -z."""
    R = np.eye(3); p = np.zeros(3)
    ps, axes = [], []
    for i, (off, Rq) in enumerate(CHAIN_R):
        p = p + R @ off
        R = R @ Rq
        ps.append(p.copy()); axes.append(R[:, 2].copy())
        R = R @ rotz(q[i])
    tip = p + R @ np.array([0, 0, -tool])
    return tip, R, ps, axes

JLIM = np.array([np.inf, 2.24, np.inf, 2.57, np.inf, 2.09, np.inf])

def ik(q0, target_p, target_R=None, tool=0.2, iters=100, wrot=0.5, tol=1e-4):
    """Damped least squares IK in arm-base frame. target_R: desired orientation of the last link frame (3x3)."""
    q = np.array(q0, float)
    for _ in range(iters):
        tip, R, ps, axes = fk_arm(q, tool)
        ep = target_p - tip
        J = np.zeros((6 if target_R is not None else 3, 7))
        for i in range(7):
            J[:3, i] = np.cross(axes[i], tip - ps[i])
            if target_R is not None:
                J[3:, i] = axes[i]
        if target_R is not None:
            Re = target_R @ R.T
            # rotation vector of Re
            ang = np.arccos(np.clip((np.trace(Re) - 1) / 2, -1, 1))
            if ang < 1e-9:
                er = np.zeros(3)
            else:
                er = ang / (2 * np.sin(ang)) * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            e = np.concatenate([ep, wrot * er])
            J[3:] *= wrot
        else:
            e = ep
        if np.linalg.norm(e) < tol:
            break
        lam = 0.01
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), e)
        n = np.max(np.abs(dq))
        if n > 0.3:
            dq *= 0.3 / n
        q = q + dq
        q = np.clip(q, -JLIM, JLIM)
    return q, np.linalg.norm(e)
