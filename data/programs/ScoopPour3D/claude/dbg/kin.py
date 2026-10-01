import numpy as np

DH = [
 (np.pi,      0.0, -0.2848,  0.0),
 (np.pi/2,    0.0, -0.0118,  np.pi),
 (np.pi/2,    0.0, -0.4208,  np.pi),
 (np.pi/2,    0.0, -0.0128,  np.pi),
 (np.pi/2,    0.0, -0.3143,  np.pi),
 (np.pi/2,    0.0,  0.0,     np.pi),
 (np.pi/2,    0.0, -0.1674,  np.pi),
]
TOOL_D = -0.0615

def _mdh(alpha, a, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0.0, a],
                     [st*ca, ct*ca, -sa, -d*sa],
                     [st*sa, ct*sa, ca, d*ca],
                     [0.0, 0.0, 0.0, 1.0]])

def fk_all(q, tool_len=0.0):
    """Return list of frames; last is the tool frame."""
    T = np.eye(4)
    frames = []
    for i, (al, a, d, off) in enumerate(DH):
        T = T @ _mdh(al, a, d, q[i] + off)
        frames.append(T.copy())
    T = T @ _mdh(np.pi, 0.0, TOOL_D, 0.0)
    if tool_len:
        Tt = np.eye(4); Tt[2, 3] = tool_len
        T = T @ Tt
    frames.append(T.copy())
    return frames

def fk(q, tool_len=0.0):
    return fk_all(q, tool_len)[-1]

def jacobian(q, tool_len=0.0):
    frames = fk_all(q, tool_len)
    pe = frames[-1][:3, 3]
    J = np.zeros((6, 7))
    Tprev = np.eye(4)
    for i in range(7):
        # joint i axis is z of frame i (after transform i)
        Ti = frames[i]
        z = Ti[:3, 2]
        p = Ti[:3, 3]
        J[:3, i] = np.cross(z, pe - p)
        J[3:, i] = z
    return J

def so3_err(Rc, Rd):
    R = Rc.T @ Rd
    w = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) * 0.5
    s = np.linalg.norm(w); c = (np.trace(R)-1)/2
    ang = np.arctan2(s, c)
    if s < 1e-8:
        return np.zeros(3) if c > 0 else np.array([np.pi,0,0])
    return Rc @ (w/s*ang)

QLIM = np.array([  # gen3: joints 1,3,5,7 continuous; 2,4,6 limited
 [-1e9, 1e9], [-2.41, 2.41], [-1e9, 1e9], [-2.66, 2.66],
 [-1e9, 1e9], [-2.23, 2.23], [-1e9, 1e9]])

def ik(target_pos, target_R, q0, tool_len=0.0, iters=200, pos_w=1.0, rot_w=0.3):
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        frames = fk_all(q, tool_len)
        T = frames[-1]
        ep = target_pos - T[:3, 3]
        er = so3_err(T[:3, :3], target_R) if target_R is not None else np.zeros(3)
        e = np.concatenate([ep*pos_w, er*rot_w])
        J = jacobian(q, tool_len)
        Jw = J.copy(); Jw[:3] *= pos_w; Jw[3:] *= rot_w
        lam = 0.05
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam**2*np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.2: dq *= 0.2/n
        q = q + dq
        q = np.clip(q, QLIM[:,0], QLIM[:,1])
        if np.linalg.norm(ep) < 1e-4 and (target_R is None or np.linalg.norm(er) < 1e-3):
            break
    frames = fk_all(q, tool_len)
    err = np.linalg.norm(target_pos - frames[-1][:3,3])
    return q, err

def rot_down(yaw=0.0):
    """EE z-axis pointing down (-Z world), x-axis rotated by yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])
