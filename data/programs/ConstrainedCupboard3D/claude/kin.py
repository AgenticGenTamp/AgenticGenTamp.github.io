"""Kinova Gen3 7-DoF forward kinematics + damped least-squares IK.

Modified-DH (Craig) table from Kinova docs:
  T_{i-1,i} = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)
"""
import numpy as np

PI = np.pi
# alpha_{i-1}, a_{i-1}, d_i, theta_offset
DH = [
    (PI,     0.0, -0.2848, 0.0),
    (PI / 2, 0.0, -0.0118, PI),
    (PI / 2, 0.0, -0.4208, PI),
    (PI / 2, 0.0, -0.0128, PI),
    (PI / 2, 0.0, -0.3143, PI),
    (PI / 2, 0.0,  0.0,    PI),
    (PI / 2, 0.0, -0.1674, PI),
]
# final fixed transform to interface plate frame
T_PLATE = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, -0.0615],
                    [0, 0, 0, 1.0]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def _tz(d):
    T = np.eye(4)
    T[2, 3] = d
    return T


def _tx(a):
    T = np.eye(4)
    T[0, 3] = a
    return T


def arm_fk(q, tool_z=0.0):
    """EE pose in the arm-base frame. tool_z: extra offset along plate z."""
    T = np.eye(4)
    for i, (al, a, d, off) in enumerate(DH):
        T = T @ _rx(al) @ _tx(a) @ _rz(q[i] + off) @ _tz(d)
    T = T @ T_PLATE
    if tool_z:
        T = T @ _tz(tool_z)
    return T


def arm_fk_all(q):
    Ts = []
    T = np.eye(4)
    for i, (al, a, d, off) in enumerate(DH):
        T = T @ _rx(al) @ _tx(a) @ _rz(q[i] + off) @ _tz(d)
        Ts.append(T.copy())
    return Ts


def jacobian(q, tool_z=0.0, eps=1e-6):
    """6xN numeric jacobian of (pos, rotvec) wrt q."""
    T0 = arm_fk(q, tool_z)
    J = np.zeros((6, 7))
    for i in range(7):
        qq = np.array(q, dtype=float)
        qq[i] += eps
        T1 = arm_fk(qq, tool_z)
        J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
        dR = T1[:3, :3] @ T0[:3, :3].T
        J[3:, i] = _log_so3(dR) / eps
    return J, T0


def _log_so3(R):
    c = (np.trace(R) - 1) / 2
    c = max(-1.0, min(1.0, c))
    th = np.arccos(c)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    return th / (2 * np.sin(th)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def quat_to_mat(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ik(T_goal, q0, tool_z=0.0, iters=200, pos_w=1.0, rot_w=0.3, q_lim=None,
       damping=1e-3, pos_tol=1e-4, rot_tol=1e-3):
    """Damped least squares IK. Returns (q, pos_err, rot_err)."""
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        J, T = jacobian(q, tool_z)
        ep = T_goal[:3, 3] - T[:3, 3]
        er = _log_so3(T_goal[:3, :3] @ T[:3, :3].T)
        if np.linalg.norm(ep) < pos_tol and np.linalg.norm(er) < rot_tol:
            break
        e = np.concatenate([pos_w * ep, rot_w * er])
        Jw = np.vstack([pos_w * J[:3], rot_w * J[3:]])
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + damping * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.3:
            dq *= 0.3 / n
        q = q + dq
        if q_lim is not None:
            q = np.clip(q, q_lim[0], q_lim[1])
    J, T = jacobian(q, tool_z)
    return q, np.linalg.norm(T_goal[:3, 3] - T[:3, 3]), np.linalg.norm(
        _log_so3(T_goal[:3, :3] @ T[:3, :3].T))


def fk_jac(q, tool_z=0.0):
    """Forward kinematics + analytic geometric Jacobian in one pass."""
    T = np.eye(4)
    axes = np.zeros((7, 3))
    origins = np.zeros((7, 3))
    for i, (al, a, d, off) in enumerate(DH):
        T = T @ _rx(al)
        if a:
            T = T @ _tx(a)
        axes[i] = T[:3, 2]
        origins[i] = T[:3, 3]
        T = T @ _rz(q[i] + off) @ _tz(d)
    T = T @ T_PLATE
    if tool_z:
        T = T @ _tz(tool_z)
    p = T[:3, 3]
    J = np.empty((6, 7))
    J[:3] = np.cross(axes, p - origins).T
    J[3:] = axes.T
    return T, J


def ik2(T_goal, q0, tool_z=0.0, iters=60, q_lim=None, pos_tol=2e-4,
        rot_tol=5e-3, rot_w=0.4):
    """Damped least squares IK using the analytic Jacobian."""
    q = np.array(q0, dtype=float)
    lam = 1e-3
    for _ in range(iters):
        T, J = fk_jac(q, tool_z)
        ep = T_goal[:3, 3] - T[:3, 3]
        er = _log_so3(T_goal[:3, :3] @ T[:3, :3].T)
        npe, nre = np.linalg.norm(ep), np.linalg.norm(er)
        if npe < pos_tol and nre < rot_tol:
            break
        e = np.concatenate([ep, rot_w * er])
        Jw = np.vstack([J[:3], rot_w * J[3:]])
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.4:
            dq *= 0.4 / n
        q += dq
        if q_lim is not None:
            np.clip(q, q_lim[0], q_lim[1], out=q)
    T, _ = fk_jac(q, tool_z)
    return q, np.linalg.norm(T_goal[:3, 3] - T[:3, 3]), np.linalg.norm(
        _log_so3(T_goal[:3, :3] @ T[:3, :3].T))
