"""Forward / inverse kinematics for the tidybot (holonomic base + Kinova Gen3 7-DOF).

World pose of the gripper "tool point" (fingertip / grasp center).
numpy only.
"""
import numpy as np

# ---- calibrated parameters (see test_pick / calibration) ----
MOUNT = np.array([0.1207, 0.0, 0.3947])   # arm base in robot-base frame (fitted)
MOUNT_YAW = 0.0
TOOL = np.array([0.0, 0.0, 0.145])    # grasp center (cube center when held) in end_effector frame


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _T(xyz, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


_P = np.pi
# (xyz, roll) of each joint origin; all rpy are pure roll here
_JOINTS = [
    ((0, 0, 0.15643), _P),
    ((0, 0.005375, -0.12838), _P / 2),
    ((0, -0.21038, -0.006375), -_P / 2),
    ((0, 0.006375, -0.21038), _P / 2),
    ((0, -0.20843, -0.006375), -_P / 2),
    ((0, 0.00017505, -0.10593), _P / 2),
    ((0, -0.10593, -0.00017505), -_P / 2),
]
_FIX = [_T(np.array(x, float), _rot('x', r)) for x, r in _JOINTS]
_EE = _T(np.array([0, 0, -0.061525]), _rot('x', _P))


def arm_chain(q, tool=None):
    """Transform of tool frame in arm-base frame, plus joint frames (for Jacobian)."""
    tool = TOOL if tool is None else tool
    T = np.eye(4)
    frames = []
    for i in range(7):
        T = T @ _FIX[i]
        frames.append(T.copy())          # joint i frame (rotation about its z)
        T = T @ _T(np.zeros(3), _rot('z', q[i]))
    T = T @ _EE
    T = T @ _T(np.asarray(tool, float), np.eye(3))
    return T, frames


def base_T(bx, by, brot, mount=None, mount_yaw=None):
    mount = MOUNT if mount is None else mount
    mount_yaw = MOUNT_YAW if mount_yaw is None else mount_yaw
    Tb = _T(np.array([bx, by, 0.0]), _rot('z', brot))
    return Tb @ _T(np.asarray(mount, float), _rot('z', mount_yaw))


def fk(bx, by, brot, q, mount=None, mount_yaw=None, tool=None):
    """Return (pos[3], R[3x3]) of the gripper tool point in world frame.
    R columns: x, y, z(approach direction, pointing out of the gripper)."""
    T = base_T(bx, by, brot, mount, mount_yaw) @ arm_chain(q, tool)[0]
    return T[:3, 3].copy(), T[:3, :3].copy()


def jacobian(bx, by, brot, q):
    """6x7 geometric Jacobian (world frame) of tool point wrt arm joints."""
    Tb = base_T(bx, by, brot)
    Tt, frames = arm_chain(q)
    p = (Tb @ Tt)[:3, 3]
    J = np.zeros((6, 7))
    for i, F in enumerate(frames):
        W = Tb @ F
        z = W[:3, 2]
        o = W[:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


# Joint limits (Kinova Gen3 7DOF, rad). Joints 1,3,5,7 are continuous.
QLO = np.array([-np.inf, -2.24, -np.inf, -2.57, -np.inf, -2.09, -np.inf])
QHI = np.array([np.inf, 2.24, np.inf, 2.57, np.inf, 2.09, np.inf])


def _rot_err(R, Rd):
    Re = Rd @ R.T
    return 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])


def ik(bx, by, brot, q0, p_des, R_des=None, z_des=None, iters=200, tol=1e-4,
       damping=0.05, wrot=0.5, qnull=None, wnull=0.02):
    """Damped least-squares IK.
    p_des: target position. R_des: full target rotation (optional).
    z_des: target approach direction only (optional; used if R_des is None).
    Returns (q, pos_err_norm)."""
    q = np.array(q0, float)
    for _ in range(iters):
        p, R = fk(bx, by, brot, q)
        J = jacobian(bx, by, brot, q)
        e = [p_des - p]
        rows = [J[:3]]
        if R_des is not None:
            e.append(wrot * _rot_err(R, R_des))
            rows.append(wrot * J[3:])
        elif z_des is not None:
            z = R[:, 2]
            e.append(wrot * np.cross(z, z_des))
            rows.append(wrot * J[3:])
        e = np.concatenate(e)
        A = np.vstack(rows)
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < tol * 10:
            break
        dq = A.T @ np.linalg.solve(A @ A.T + damping ** 2 * np.eye(A.shape[0]), e)
        if qnull is not None:
            N = np.eye(7) - np.linalg.pinv(A) @ A
            dq += N @ (wnull * (np.asarray(qnull) - q))
        n = np.abs(dq).max()
        if n > 0.3:
            dq *= 0.3 / n
        q = np.clip(q + dq, QLO, QHI)
    p, _ = fk(bx, by, brot, q)
    return q, np.linalg.norm(p_des - p)


Q_READY = np.array([0.0, 0.26, np.pi, -2.27, 0.0, 0.96, np.pi / 2])  # gripper down, in front


def ik_multi(bx, by, br, q0, p_des, R_des=None, z_des=None, n_rand=6, seed=0, **kw):
    """IK with restarts (current q, ready pose, random). Returns best (q, err),
    preferring solutions close to q0 among those with err < 2mm."""
    rng = np.random.default_rng(seed)
    starts = [np.asarray(q0, float), Q_READY.copy()]
    # ready pose rotated toward target
    Tb = base_T(bx, by, br)
    loc = np.linalg.solve(Tb, np.r_[p_des, 1.0])[:3]
    qr = Q_READY.copy(); qr[0] = np.arctan2(loc[1], loc[0]); starts.append(qr)
    for _ in range(n_rand):
        starts.append(qr + rng.normal(size=7) * 0.5)
    best = None
    for s in starts:
        q, e = ik(bx, by, br, s, p_des, R_des=R_des, z_des=z_des, qnull=qr, **kw)
        # orientation error also matters
        _, R = fk(bx, by, br, q)
        eo = 0.0 if R_des is None else np.linalg.norm(_rot_err(R, R_des))
        if R_des is None and z_des is not None:
            eo = np.linalg.norm(np.cross(R[:, 2], z_des))
        cost = e + 0.05 * eo + (0.001 * np.linalg.norm(q - q0) if e < 0.002 else 0)
        if best is None or cost < best[0]:
            best = (cost, q, e)
    return best[1], best[2]


# ---- joint servo (identified from env): per-step joint displacement
#   dq_k = 0.059 * u_k - 0.59 * dq_{k-1}   (u = vel-target action a[11:18], dq in rad/step, dt=0.1s)
#   steady state: joint vel = 0.3645*u rad/s ; hard cap 4 rad/s (0.4 rad/step).
#   deltas a[3:10] behave similarly with gain 0.394/step (steady 0.25*delta per step).
def servo_action(q, qdot, q_target, gain=0.8, vmax=0.35, grip=0.0):
    """18-d action driving arm joints to q_target (near-deadbeat, ~5-10 steps)."""
    a = np.zeros(18, np.float32)
    y = np.clip(gain * (np.asarray(q_target) - q), -vmax, vmax)
    a[11:18] = (y + 0.59 * np.asarray(qdot) * 0.1) / 0.059
    a[10] = grip
    return a


if __name__ == '__main__':
    home = [0, -0.349, 3.142, -2.548, 0, -0.873, 1.571]
    p, R = fk(0, 0, 0, home)
    print('home tool pos', p.round(4))
    print('home R\n', R.round(3))
