import numpy as np

def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# Kinova Gen3 7-dof (mujoco menagerie / URDF), each joint about local z
_LINKS = [
    (np.array([0, 0, 0.15643]), _rx(np.pi)),
    (np.array([0, 0.005375, -0.12838]), _rx(np.pi / 2)),
    (np.array([0, -0.21038, -0.006375]), _rx(-np.pi / 2)),
    (np.array([0, 0.006375, -0.21038]), _rx(np.pi / 2)),
    (np.array([0, -0.20843, -0.006375]), _rx(-np.pi / 2)),
    (np.array([0, 0.00017505, -0.10593]), _rx(np.pi / 2)),
    (np.array([0, -0.10593, -0.00017505]), _rx(-np.pi / 2)),
]
FLANGE = np.array([0, 0, -0.0615])  # bracelet -> flange (z axis of bracelet points toward -z)

class Kin:
    def __init__(self, mount=(0.12, 0.0, 0.36), tool=0.15):
        self.mount = np.array(mount, dtype=float)
        self.tool = tool

    def fk(self, base, q):
        """base=(x,y,th), q = 7 joints. returns tool point pos (world), rotation of flange frame."""
        x, y, th = base
        R = _rz(th)
        p = np.array([x, y, 0.0]) + R @ self.mount
        for i in range(7):
            t, Rl = _LINKS[i]
            p = p + R @ t
            R = R @ Rl @ _rz(q[i])
        # tool frame: bracelet z axis points "backward"; tool extends along -z of bracelet
        p_tool = p + R @ np.array([0, 0, -(0.0615 + self.tool)])
        return p_tool, R

    def jac(self, base, q, eps=1e-5):
        p0, R0 = self.fk(base, q)
        J = np.zeros((6, 7))
        for i in range(7):
            dq = np.array(q, dtype=float).copy(); dq[i] += eps
            p1, R1 = self.fk(base, dq)
            J[:3, i] = (p1 - p0) / eps
            dR = R1 @ R0.T
            w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2
            J[3:, i] = w / eps
        return J, p0, R0

    def ik(self, base, q0, p_target, R_target=None, iters=100, tol=1e-4, lam=0.02, wrot=0.3):
        q = np.array(q0, dtype=float).copy()
        for _ in range(iters):
            J, p, R = self.jac(base, q)
            e = np.zeros(6)
            e[:3] = p_target - p
            if R_target is not None:
                dR = R_target @ R.T
                e[3:] = wrot * 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
                Jw = J.copy(); Jw[3:] *= wrot
            else:
                Jw = J[:3]; e = e[:3]
            if np.linalg.norm(e) < tol:
                break
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam**2 * np.eye(Jw.shape[0]), e)
            n = np.abs(dq).max()
            if n > 0.3: dq *= 0.3 / n
            q += dq
        return q, np.linalg.norm(e)
