"""Kinova Gen3 forward kinematics + simple IK (numpy only)."""
import numpy as np

def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def _T(xyz, rpy):
    T = np.eye(4); T[:3, :3] = _rpy(*rpy); T[:3, 3] = xyz; return T

PI = np.pi
JOINTS = [
    _T([0, 0, 0.15643], [PI, 0, 0]),
    _T([0, 0.005375, -0.12838], [PI / 2, 0, 0]),
    _T([0, -0.21038, -0.006375], [-PI / 2, 0, 0]),
    _T([0, 0.006375, -0.21038], [PI / 2, 0, 0]),
    _T([0, -0.20843, -0.006375], [-PI / 2, 0, 0]),
    _T([0, 0.00017505, -0.10593], [PI / 2, 0, 0]),
    _T([0, -0.10593, -0.00017505], [-PI / 2, 0, 0]),
]
T_EE = _T([0, 0, -0.061525], [PI, 0, 0])

def _Rz(q):
    c, s = np.cos(q), np.sin(q)
    T = np.eye(4); T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c; return T

def fk_arm(q, tool=0.0):
    """Pose of tool point in arm base frame. tool = offset along EE z."""
    T = np.eye(4)
    for i in range(7):
        T = T @ JOINTS[i] @ _Rz(q[i])
    T = T @ T_EE
    if tool:
        T = T @ _T([0, 0, tool], [0, 0, 0])
    return T

# ---- world <-> arm base; mount params (calibrated) ----
MOUNT = np.array([0.1203, 0.0, 0.3943])  # calibrated via held-cube fit
TOOL = 0.15                           # distance from EE link along its z to tool point
Q_LO = np.array([-1e9, -2.24, -1e9, -2.57, -1e9, -2.09, -1e9])
Q_HI = -Q_LO

def arm_base_T(base):
    x, y, th = base
    T = _Rz(th); p = np.array([x, y, 0]) + T[:3, :3] @ MOUNT
    T[:3, 3] = p
    return T

def fk_world(base, q, tool=None):
    return arm_base_T(base) @ fk_arm(q, TOOL if tool is None else tool)

def rot_err(Rc, Rd):
    """Legacy sin-based rotation error (validated behaviour; ambiguous at pi)."""
    Re = Rd @ Rc.T
    return np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) * 0.5


def rot_err_log(Rc, Rd):
    """Rotation vector (log map) of Rd @ Rc.T, robust near pi."""
    Re = Rd @ Rc.T
    v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) * 0.5
    c = np.clip((np.trace(Re) - 1) * 0.5, -1.0, 1.0)
    ang = np.arccos(c)
    s = np.sin(ang)
    if s > 1e-3:
        return v * (ang / s)
    if c > 0:
        return v
    # near pi: axis from symmetric part
    B = (Re + np.eye(3)) * 0.5
    k = int(np.argmax(np.diag(B)))
    n = B[:, k] / np.sqrt(max(B[k, k], 1e-12))
    if np.dot(n, v) < 0:
        n = -n
    return np.pi * n

def ik(base, q0, p_des, R_des=None, tool=None, iters=100, wrot=0.5, tol=1e-4):
    """DLS IK in world frame. R_des None -> position only. Returns q, err."""
    q = np.array(q0, dtype=float).copy()
    Tb = arm_base_T(base)
    for it in range(iters):
        T = Tb @ fk_arm(q, TOOL if tool is None else tool)
        e = p_des - T[:3, 3]
        if R_des is not None:
            e = np.concatenate([e, wrot * rot_err(T[:3, :3], R_des)])
        if np.linalg.norm(e) < tol:
            break
        J = np.zeros((len(e), 7)); h = 1e-6
        for j in range(7):
            dq = q.copy(); dq[j] += h
            Tj = Tb @ fk_arm(dq, TOOL if tool is None else tool)
            ej = p_des - Tj[:3, 3]
            if R_des is not None:
                ej = np.concatenate([ej, wrot * rot_err(Tj[:3, :3], R_des)])
            J[:, j] = -(ej - e) / h
        lam = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(len(e)), e)
        n = np.max(np.abs(dq))
        if n > 0.3: dq *= 0.3 / n
        q = np.clip(q + dq, Q_LO, Q_HI)
    return q, np.linalg.norm(e)

def R_down(yaw):
    """Tool z pointing down, tool x along world yaw direction."""
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([c, s, 0]); z = np.array([0, 0, -1.0]); y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rotvec(v):
    v = np.asarray(v, float); a = np.linalg.norm(v)
    if a < 1e-12:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def R_tilt(yaw, d, alpha):
    """R_down(yaw) tilted by alpha so tool z leans toward horizontal direction d."""
    d = np.array([d[0], d[1], 0.0]); d /= np.linalg.norm(d)
    axis = np.cross([0, 0, -1.0], d)   # rotating -z toward d
    return rotvec(axis * alpha) @ R_down(yaw)
