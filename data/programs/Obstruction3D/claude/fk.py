import numpy as np

# Kinova Gen3 7-DOF URDF chain (kortex_description GEN3_URDF_V12)
# Each entry: (xyz, rpy) of the joint origin relative to previous link frame; axis is +z.
CHAIN = [
    ((0.0, 0.0, 0.15643), (-np.pi, 0.0, 0.0)),
    ((0.0, 0.005375, -0.12838), (np.pi/2, 0.0, 0.0)),
    ((0.0, -0.21038, -0.006375), (-np.pi/2, 0.0, 0.0)),
    ((0.0, 0.006375, -0.21038), (np.pi/2, 0.0, 0.0)),
    ((0.0, -0.20843, -0.006375), (-np.pi/2, 0.0, 0.0)),
    ((0.0, 0.00017505, -0.10593), (np.pi/2, 0.0, 0.0)),
    ((0.0, -0.10593, -0.00017505), (-np.pi/2, 0.0, 0.0)),
]
EE_ORIGIN = ((0.0, 0.0, -0.06153), (np.pi, 0.0, 0.0))


def rpy_to_mat(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def tf(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_mat(*rpy)
    T[:3, 3] = xyz
    return T


_FIXED = [tf(*c) for c in CHAIN]
_EE = tf(*EE_ORIGIN)


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4)
    T[0, 0] = c; T[0, 1] = -s; T[1, 0] = s; T[1, 1] = c
    return T


def fk_frames(q):
    """Return list of link frames (base_link frame) and ee frame."""
    T = np.eye(4)
    frames = []
    for i in range(7):
        T = T @ _FIXED[i] @ rotz(q[i])
        frames.append(T.copy())
    T = T @ _EE
    frames.append(T.copy())
    return frames


def fk_ee(q, tool_z=0.0):
    T = fk_frames(q)[-1]
    if tool_z:
        T = T @ tf((0, 0, tool_z), (0, 0, 0))
    return T


def base_tf(bx, by, brot, bz=0.0):
    T = np.eye(4)
    c, s = np.cos(brot), np.sin(brot)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T[:3, 3] = (bx, by, bz)
    return T


def jacobian(q, tool_z=0.0):
    """6xN space Jacobian of ee (in base_link frame): [v; w]."""
    frames = fk_frames(q)
    Tee = frames[-1]
    if tool_z:
        Tee = Tee @ tf((0, 0, tool_z), (0, 0, 0))
    p_ee = Tee[:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        Ti = frames[i]
        z = Ti[:3, 2]
        p = Ti[:3, 3]
        J[:3, i] = np.cross(z, p_ee - p)
        J[3:, i] = z
    return J
