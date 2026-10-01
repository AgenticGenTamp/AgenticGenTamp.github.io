import numpy as np

# PR2 left arm chain, in torso_lift_link frame
# (translation, axis) per joint
LINKS = [
    ((0.0, 0.188, 0.0), 'z'),   # l_shoulder_pan
    ((0.1, 0.0, 0.0), 'y'),     # l_shoulder_lift
    ((0.0, 0.0, 0.0), 'x'),     # l_upper_arm_roll
    ((0.4, 0.0, 0.0), 'y'),     # l_elbow_flex
    ((0.0, 0.0, 0.0), 'x'),     # l_forearm_roll
    ((0.321, 0.0, 0.0), 'y'),   # l_wrist_flex
    ((0.0, 0.0, 0.0), 'x'),     # l_wrist_roll
]
TOOL = np.array([0.18011, 0.0, 0.0])
TORSO_OFF = np.array([-0.05, 0.0, 0.990775])
LIMITS = np.array([[-0.7146, 2.2854], [-0.5236, 1.3963], [-0.8, 3.9],
                   [-2.3213, 0.0], [-1e9, 1e9], [-2.094, 0.0], [-1e9, 1e9]])

def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == 'x':
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    if axis == 'y':
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def fk_arm(q, torso_lift=0.0):
    """Return (pos, R) of tool frame in base_link frame."""
    p = TORSO_OFF + np.array([0.0, 0.0, torso_lift])
    R = np.eye(3)
    frames = []
    for i, ((t, ax)) in enumerate(LINKS):
        p = p + R @ np.array(t)
        R = R @ rot(ax, q[i])
        frames.append((p.copy(), R.copy()))
    p = p + R @ TOOL
    return p, R, frames

def fk_world(base, q, torso_lift=0.0):
    """base = (x,y,yaw). Returns tool pos, R in world."""
    p, R, frames = fk_arm(q, torso_lift)
    c, s = np.cos(base[2]), np.sin(base[2])
    Rb = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return Rb @ p + np.array([base[0], base[1], 0.0]), Rb @ R, frames

def jacobian(q, torso_lift=0.0):
    p_e, R_e, frames = fk_arm(q, torso_lift)
    J = np.zeros((6, 7))
    for i, (p_i, R_i) in enumerate(frames):
        ax = LINKS[i][1]
        a = R_i @ ({'x': np.array([1.,0,0]), 'y': np.array([0,1.,0]), 'z': np.array([0,0,1.])}[ax])
        # note: R_i already includes joint i rotation, axis unchanged by own rotation
        J[:3, i] = np.cross(a, p_e - p_i)
        J[3:, i] = a
    return J

def so3_error(R_cur, R_des):
    E = R_des @ R_cur.T
    w = np.array([E[2,1]-E[1,2], E[0,2]-E[2,0], E[1,0]-E[0,1]])
    c = (np.trace(E) - 1) / 2
    c = np.clip(c, -1, 1)
    th = np.arccos(c)
    sn = np.sin(th)
    if sn < 1e-8:
        return w * 0.5
    return w * (th / (2 * sn))

Q_NOM = np.array([0.6772, -0.3431, 1.2, -1.4669, 1.2422, -1.9544, 2.2225])


def ik(target_p, target_R, q0, torso_lift=0.0, iters=200, tol=1e-4, q_nom=None,
       kn=0.06):
    """IK in base frame, with a null-space posture bias."""
    q = np.array(q0, dtype=float)
    if q_nom is None:
        q_nom = Q_NOM
    for it in range(iters):
        p, R, _ = fk_arm(q, torso_lift)
        ep = target_p - p
        er = so3_error(R, target_R)
        e = np.concatenate([ep, er])
        if np.linalg.norm(ep) < tol and np.linalg.norm(er) < 1e-3:
            return q, True
        J = jacobian(q, torso_lift)
        lam = 0.05
        Jp = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(6))
        dq = Jp @ e
        if kn and (np.linalg.norm(ep) > 0.02 or np.linalg.norm(er) > 0.05):
            dn = q_nom - q
            dn[4] = (dn[4] + np.pi) % (2 * np.pi) - np.pi
            dn[6] = (dn[6] + np.pi) % (2 * np.pi) - np.pi
            dq = dq + (np.eye(7) - Jp @ J) @ (kn * dn)
        n = np.linalg.norm(dq)
        if n > 0.2:
            dq = dq * (0.2 / n)
        q = q + dq
        q = np.clip(q, LIMITS[:,0], LIMITS[:,1])
    p, R, _ = fk_arm(q, torso_lift)
    ok = np.linalg.norm(target_p - p) < 1e-3 and np.linalg.norm(so3_error(R, target_R)) < 1e-2
    return q, ok
