"""PR2 left-arm forward kinematics + IK (numpy only)."""
import numpy as np

JOINT_LOW = np.array([-0.7146, -0.5236, -0.8, -2.3213, -np.inf, -2.094, -np.inf])
JOINT_HIGH = np.array([2.2854, 1.3963, 3.9, 0.0, np.inf, 0.0, np.inf])
CONT = [4, 6]

# link offsets (URDF PR2)
TORSO_XY = np.array([-0.05, 0.188])
TORSO_Z = 0.99068
L_UP = 0.4
L_FORE = 0.321
L_TOOL = 0.18


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def arm_fk(q, torso_z=TORSO_Z, tool=L_TOOL, links=False):
    """Tool frame pose (p, R) in the robot base frame (base at origin, rot 0)."""
    p = np.array([TORSO_XY[0], TORSO_XY[1], torso_z])
    R = _rz(q[0])
    pts = [p.copy()]
    p = p + R @ np.array([0.1, 0.0, 0.0])
    R = R @ _ry(q[1]) @ _rx(q[2])
    pts.append(p.copy())
    p = p + R @ np.array([L_UP, 0.0, 0.0])
    R = R @ _ry(q[3]) @ _rx(q[4])
    pts.append(p.copy())
    p = p + R @ np.array([L_FORE, 0.0, 0.0])
    R = R @ _ry(q[5]) @ _rx(q[6])
    pts.append(p.copy())
    p = p + R @ np.array([tool, 0.0, 0.0])
    pts.append(p.copy())
    if links:
        return p, R, pts
    return p, R


def world_fk(q, base, torso_z=TORSO_Z, tool=L_TOOL):
    p, R = arm_fk(q, torso_z, tool)
    Rb = _rz(base[2])
    return Rb @ p + np.array([base[0], base[1], 0.0]), Rb @ R


def pose_err(q, base, tgt_p, tgt_R, torso_z=TORSO_Z, tool=L_TOOL, w_rot=1.0):
    p, R = world_fk(q, base, torso_z, tool)
    ep = p - tgt_p
    # rotation error as axis-angle vector
    Re = R @ tgt_R.T
    v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    s = np.linalg.norm(v)
    c = (np.trace(Re) - 1) / 2.0
    ang = np.arctan2(s / 2.0, c)
    er = v / (s + 1e-12) * ang if s > 1e-9 else (np.zeros(3) if c > 0 else np.array([np.pi, 0, 0]))
    return np.concatenate([ep, w_rot * er])


def clip_q(q):
    q = np.array(q, dtype=float)
    for i in range(7):
        if i in CONT:
            q[i] = (q[i] + np.pi) % (2 * np.pi) - np.pi
        else:
            q[i] = min(max(q[i], JOINT_LOW[i]), JOINT_HIGH[i])
    return q


def ik(tgt_p, tgt_R, base, q0, torso_z=TORSO_Z, tool=L_TOOL, iters=200, seeds=8,
       rng=None, w_rot=0.3, q_ref=None, w_ref=0.0):
    """Damped least squares IK with random restarts. Returns (q, err)."""
    if rng is None:
        rng = np.random.default_rng(0)
    best, best_e = None, 1e9
    cands = [np.array(q0, dtype=float)]
    for _ in range(seeds - 1):
        lo = np.where(np.isfinite(JOINT_LOW), JOINT_LOW, -np.pi)
        hi = np.where(np.isfinite(JOINT_HIGH), JOINT_HIGH, np.pi)
        cands.append(rng.uniform(lo, hi))
    for q in cands:
        q = clip_q(q)
        lam = 0.1
        for _ in range(iters):
            e = pose_err(q, base, tgt_p, tgt_R, torso_z, tool, w_rot)
            J = np.zeros((6, 7))
            for i in range(7):
                dq = np.zeros(7); dq[i] = 1e-5
                J[:, i] = (pose_err(q + dq, base, tgt_p, tgt_R, torso_z, tool, w_rot) - e) / 1e-5
            H = J.T @ J + lam * np.eye(7)
            g = J.T @ e
            if w_ref > 0 and q_ref is not None:
                H += w_ref * np.eye(7)
                g += w_ref * (q - np.asarray(q_ref))
            step = np.linalg.solve(H, -g)
            n = np.linalg.norm(step)
            if n > 0.5:
                step *= 0.5 / n
            q = clip_q(q + step)
            if n < 1e-7:
                break
        e = pose_err(q, base, tgt_p, tgt_R, torso_z, tool, w_rot)
        sc = np.linalg.norm(e[:3]) + 0.05 * np.linalg.norm(e[3:])
        if sc < best_e:
            best_e, best = sc, q.copy()
    return best, best_e


def grasp_R(yaw):
    """Tool orientation: x-axis = approach dir (horizontal yaw), z-axis = world up."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
