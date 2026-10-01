"""Shared control helpers: FK/IK-based Cartesian motion for the tidybot arm."""
import numpy as np
import kin

MOUNT = np.array([0.1197, 0.0003, 0.3952])   # arm base in robot-base frame
TOOL = 0.1009                                 # grasp point offset along tool z
LIM = np.array([[-1e6, -2.20, -1e6, -2.54, -1e6, -2.07, -1e6],
                [1e6, 2.20, 1e6, 2.54, 1e6, 2.07, 1e6]])
SOFT = np.array([[-6.0, -1.80, -6.0, -2.40, -6.0, -1.90, -6.0],
                 [6.0, 1.80, 6.0, 2.40, 6.0, 1.90, 6.0]])
BASE_GAIN = 0.87
ARM_GAIN = 1.0
J_NAMES = ['pos_arm_joint%d' % (i + 1) for i in range(7)]


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def read_robot(state):
    r = state.get_object_from_name('robot')
    q = np.array([state.get(r, n) for n in J_NAMES])
    b = np.array([state.get(r, 'pos_base_x'), state.get(r, 'pos_base_y'),
                  state.get(r, 'pos_base_rot')])
    return q, b


def pose_from(pos, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def world_to_arm(T_world, b):
    Rb = rotz(b[2])
    T = np.eye(4)
    T[:3, :3] = Rb.T @ T_world[:3, :3]
    T[:3, 3] = Rb.T @ (T_world[:3, 3] - np.array([b[0], b[1], 0.0])) - MOUNT
    return T


def arm_to_world(T_arm, b):
    Rb = rotz(b[2])
    T = np.eye(4)
    T[:3, :3] = Rb @ T_arm[:3, :3]
    T[:3, 3] = Rb @ (T_arm[:3, 3] + MOUNT) + np.array([b[0], b[1], 0.0])
    return T


def ee_world(q, b, tool=TOOL):
    return arm_to_world(kin.arm_fk(q, tool), b)


def roll_pose(T, ang):
    """Rotate a pose about its own tool z axis."""
    c, s = np.cos(ang), np.sin(ang)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    out = T.copy()
    out[:3, :3] = T[:3, :3] @ Rz
    return out


def ik_multi(T_arm, q0, tries=16, tool=TOOL, rng=None, near=None, rolls=(0.0,)):
    """IK with random restarts (and optional free roll about the tool axis)."""
    if rng is None:
        rng = np.random.default_rng(0)
    ref = q0 if near is None else near
    best = None
    for ang in rolls:
        T = T_arm if ang == 0.0 else roll_pose(T_arm, ang)
        for k in range(tries):
            s = q0 if k == 0 else np.clip(q0 + rng.uniform(-2.0, 2.0, 7), LIM[0], LIM[1])
            qd, ep, er = kin.ik2(T, s, tool_z=tool, q_lim=LIM, iters=80)
            if ep > 0.004 or er > 0.05:
                continue
            sc = np.max(np.abs(qd - ref))
            if best is None or sc < best[0]:
                best = (sc, qd, T)
            if best[0] < 0.6:
                break
        if best is not None and best[0] < 0.6:
            break
    return None if best is None else best[1]


def ik_multi_pose(T_arm, q0, tries=16, tool=TOOL, rng=None, near=None, rolls=(0.0,)):
    """Like ik_multi but also returns the (possibly rolled) pose used."""
    if rng is None:
        rng = np.random.default_rng(0)
    ref = q0 if near is None else near
    best = None
    for ang in rolls:
        T = T_arm if ang == 0.0 else roll_pose(T_arm, ang)
        for k in range(tries):
            s = q0 if k == 0 else np.clip(q0 + rng.uniform(-2.0, 2.0, 7), LIM[0], LIM[1])
            qd, ep, er = kin.ik2(T, s, tool_z=tool, q_lim=LIM, iters=80)
            if ep > 0.004 or er > 0.05:
                continue
            sc = np.max(np.abs(qd - ref))
            if best is None or sc < best[0]:
                best = (sc, qd, T)
            if best[0] < 0.6:
                break
        if best is not None and best[0] < 0.6:
            break
    return (None, None) if best is None else (best[1], best[2])


def slerp_R(R0, R1, t):
    dR = R1 @ R0.T
    w = kin._log_so3(dR) * t
    th = np.linalg.norm(w)
    if th < 1e-12:
        return R0.copy()
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K) @ R0


def cartesian_traj(q_start, T0_arm, T1_arm, tool=TOOL, dp=0.02, dth=0.06,
                   max_jstep=0.022):
    """Joint-space trajectory following a straight Cartesian line."""
    dist = np.linalg.norm(T1_arm[:3, 3] - T0_arm[:3, 3])
    ang = np.linalg.norm(kin._log_so3(T1_arm[:3, :3] @ T0_arm[:3, :3].T))
    n = max(2, int(max(dist / dp, ang / dth)) + 1)
    q = np.array(q_start, dtype=float)
    traj = []
    for i in range(1, n + 1):
        t = i / n
        p = T0_arm[:3, 3] * (1 - t) + T1_arm[:3, 3] * t
        R = slerp_R(T0_arm[:3, :3], T1_arm[:3, :3], t)
        T = pose_from(p, R)
        qn, ep, er = kin.ik2(T, q, tool_z=tool, q_lim=LIM, iters=40)
        if ep > 0.01:
            return None
        # subdivide big joint jumps
        step = np.max(np.abs(qn - q))
        if step > max_jstep:
            m = int(np.ceil(step / max_jstep))
            for j in range(1, m + 1):
                traj.append(q + (qn - q) * (j / m))
        else:
            traj.append(qn.copy())
        q = qn
    return traj
