import numpy as np

DH_ALPHA = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi], dtype=float)
DH_D = np.array([-0.2848, -0.0118, -0.4208, -0.0128, -0.3143, 0.0, -0.1674], dtype=float)
DH_TH = np.array([0.0, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi], dtype=float)
TOOL_OFFSET = 0.1267
# base -> frame0 : rotation of pi about X
T_B0 = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=float)
LIM_LO = np.array([-np.inf, -2.2497, -np.inf, -2.5796, -np.inf, -2.0996, -np.inf])
LIM_HI = np.array([ np.inf,  2.2497,  np.inf,  2.5796,  np.inf,  2.0996,  np.inf])
CONT = np.array([True, False, True, False, True, False, True])


def _frames(q, tool_z=0.0):
    T = T_B0.copy()
    out = [T.copy()]
    for i in range(7):
        th = q[i] + DH_TH[i]
        ca, sa = np.cos(DH_ALPHA[i]), np.sin(DH_ALPHA[i])
        ct, st = np.cos(th), np.sin(th)
        A = np.array([[ct, -st*ca,  st*sa, 0.0],
                      [st,  ct*ca, -ct*sa, 0.0],
                      [0.0,    sa,     ca, DH_D[i]],
                      [0.0,   0.0,    0.0, 1.0]])
        T = T @ A
        out.append(T.copy())
    if tool_z:
        Tt = np.eye(4); Tt[2,3] = tool_z
        out.append(T @ Tt)
    else:
        out.append(T.copy())
    return out


def fk(q, tool_z=0.0):
    return _frames(q, tool_z)[-1]


def fk_pos(q, tool_z=0.0):
    return _frames(q, tool_z)[-1][:3, 3]


def jacobian(q, tool_z=0.0):
    fr = _frames(q, tool_z)
    pe = fr[-1][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        Ti = fr[i+1]           # frame after joint i
        z = Ti[:3, 2]
        # joint i rotates about z of frame i (i.e. fr[i] z-axis in DH convention)
        zi = fr[i][:3, 2]
        pi_ = fr[i][:3, 3]
        J[:3, i] = np.cross(zi, pe - pi_)
        J[3:, i] = zi
    return J


def wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def rot_err(Rc, Rd):
    Re = Rd @ Rc.T
    w = np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]])
    s = np.linalg.norm(w)
    c = (np.trace(Re)-1.0)/2.0
    ang = np.arctan2(s/2.0, c)
    if s < 1e-9:
        return np.zeros(3) if c > 0 else np.array([np.pi,0,0])
    return w/s*ang


def clamp(q):
    q = q.copy()
    for i in range(7):
        if CONT[i]:
            q[i] = wrap(q[i])
        else:
            q[i] = min(max(q[i], LIM_LO[i]), LIM_HI[i])
    return q


def ik_local(target_pos, R_des, q_init, tool_z=TOOL_OFFSET, iters=200,
             pos_tol=1e-4, rot_tol=2e-3, w_rot=1.0, q_bias=None, k_null=0.02,
             free_yaw=False):
    """Damped least squares IK starting from q_init, staying local."""
    q = np.array(q_init, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    best = (1e9, q.copy())
    for it in range(iters):
        T = fk(q, tool_z)
        ep = target_pos - T[:3, 3]
        if R_des is not None:
            er = rot_err(T[:3, :3], R_des)
            if free_yaw:
                # allow free rotation about the tool z axis
                zt = T[:3, 2]
                er = er - np.dot(er, zt)*zt
        else:
            er = np.zeros(3)
        cost = np.linalg.norm(ep) + 0.1*np.linalg.norm(er)
        if cost < best[0]:
            best = (cost, q.copy())
        if np.linalg.norm(ep) < pos_tol and np.linalg.norm(er) < rot_tol:
            return q, True
        e = np.concatenate([np.clip(ep, -0.05, 0.05), w_rot*np.clip(er, -0.2, 0.2)])
        J = jacobian(q, tool_z)
        lam = 0.03
        dq = J.T @ np.linalg.solve(J @ J.T + lam*lam*np.eye(6), e)
        if q_bias is not None:
            N = np.eye(7) - J.T @ np.linalg.solve(J @ J.T + lam*lam*np.eye(6), J)
            dq = dq + N @ (k_null*(np.asarray(q_bias)-q))
        n = np.linalg.norm(dq)
        if n > 0.25:
            dq = dq*(0.25/n)
        q = clamp(q + dq)
    T = fk(best[1], tool_z)
    ok = np.linalg.norm(target_pos-T[:3,3]) < 3e-3
    return best[1], ok
