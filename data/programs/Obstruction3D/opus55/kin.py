import numpy as np
# Kinova Gen3 7DOF chain (kortex gen3.urdf); calibrated mount offset & tool offset.
JOINTS = [
    ((0, 0, 0.15643), (np.pi, 0, 0)),
    ((0, 0.005375, -0.12838), (np.pi/2, 0, 0)),
    ((0, -0.21038, -0.006375), (-np.pi/2, 0, 0)),
    ((0, 0.006375, -0.21038), (np.pi/2, 0, 0)),
    ((0, -0.20843, -0.006375), (-np.pi/2, 0, 0)),
    ((0, 0.00017505, -0.10593), (np.pi/2, 0, 0)),
    ((0, -0.10593, -0.00017505), (-np.pi/2, 0, 0)),
]
MOUNT = (0.12, 0.0, -0.0052)
TOOL = 0.12  # grasp frame is TOOL beyond end_effector_link along its z

def rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])

def T(xyz, rpy):
    M = np.eye(4); M[:3,:3] = rpy2R(*rpy); M[:3,3] = xyz; return M

def Rz(q):
    M = np.eye(4); c, s = np.cos(q), np.sin(q); M[0,0]=c; M[0,1]=-s; M[1,0]=s; M[1,1]=c; return M

JT = [T(*j) for j in JOINTS]
EET = T((0, 0, -0.061525 - TOOL), (np.pi, 0, 0))  # combined end effector + tool

def base_T(base):
    bx, by, br = base
    M = Rz(br); M[0,3] = bx; M[1,3] = by
    return M @ T(MOUNT, (0, 0, 0))

def fk(base, q):
    M = base_T(base)
    for i in range(7):
        M = M @ JT[i] @ Rz(q[i])
    return M @ EET

def fk_all(base, q):
    """Return list of frames: mount, after each joint, tool."""
    M = base_T(base); out = [M]
    for i in range(7):
        M = M @ JT[i] @ Rz(q[i]); out.append(M)
    out.append(M @ EET)
    return out

def jac(base, q):
    """Geometric Jacobian (6x7) of tool point: rows pos, rot."""
    Ms = fk_all(base, q)
    p = Ms[-1][:3,3]
    J = np.zeros((6,7))
    for i in range(7):
        z = Ms[i+1][:3,2]; o = Ms[i+1][:3,3]
        J[:3,i] = np.cross(z, p - o); J[3:,i] = z
    return J, Ms[-1]

def rot_err(Rc, Rt):
    E = Rt @ Rc.T
    return 0.5*np.array([E[2,1]-E[1,2], E[0,2]-E[2,0], E[1,0]-E[0,1]])

def ik(base, q0, pos, R, iters=100, w_ori=0.5, tol=1e-5):
    q = np.array(q0, dtype=float)
    err = 1e9
    for it in range(iters):
        J, M = jac(base, q)
        e = np.concatenate([pos - M[:3,3], w_ori*rot_err(M[:3,:3], R)])
        err = np.linalg.norm(e)
        if err < tol: break
        Jw = J.copy(); Jw[3:] *= w_ori
        dq = Jw.T @ np.linalg.solve(Jw@Jw.T + 1e-4*np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.4: dq *= 0.4/n
        q += dq
    return q, err

def down_R(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0],[s, -c, 0],[0, 0, -1.0]])
