import numpy as np
# Kinova Gen3 7DOF (kortex_description gen3.urdf) joint origins: xyz, rpy; all axes z
JOINTS = [
    ((0, 0, 0.15643), (np.pi, 0, 0)),
    ((0, 0.005375, -0.12838), (np.pi/2, 0, 0)),
    ((0, -0.21038, -0.006375), (-np.pi/2, 0, 0)),
    ((0, 0.006375, -0.21038), (np.pi/2, 0, 0)),
    ((0, -0.20843, -0.006375), (-np.pi/2, 0, 0)),
    ((0, 0.00017505, -0.10593), (np.pi/2, 0, 0)),
    ((0, -0.10593, -0.00017505), (-np.pi/2, 0, 0)),
]
EE = ((0, 0, -0.061525), (np.pi, 0, 0))

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
EET = T(*EE)

def fk(base, q, base_z=0.0, tool=0.0):
    bx, by, br = base
    M = Rz(br); M[0,3]=bx; M[1,3]=by; M[2,3]=base_z
    for i in range(7):
        M = M @ JT[i] @ Rz(q[i])
    M = M @ EET
    if tool:
        M = M.copy(); M[:3,3] = M[:3,3] + M[:3,2]*tool
    return M

def pose_err(M, pos, R):
    ep = pos - M[:3,3]
    Rc = M[:3,:3]
    # orientation error (axis-angle approx)
    E = R @ Rc.T
    eo = 0.5*np.array([E[2,1]-E[1,2], E[0,2]-E[2,0], E[1,0]-E[0,1]])
    return np.concatenate([ep, eo])

def ik(base, q0, pos, R, base_z=0.0, tool=0.0, iters=200, w_ori=0.3, tol=1e-4):
    q = np.array(q0, dtype=float)
    for it in range(iters):
        M = fk(base, q, base_z, tool)
        e = pose_err(M, pos, R); e[3:] *= w_ori
        if np.linalg.norm(e) < tol: break
        J = np.zeros((6,7)); h=1e-6
        for i in range(7):
            dq = q.copy(); dq[i]+=h
            e2 = pose_err(fk(base, dq, base_z, tool), pos, R); e2[3:]*=w_ori
            J[:,i] = -(e2-e)/h
        lam = 1e-3
        dq = J.T @ np.linalg.solve(J@J.T + lam*np.eye(6), e)
        n = np.linalg.norm(dq)
        if n > 0.3: dq *= 0.3/n
        q += dq
    return q, np.linalg.norm(e)

def down_R(yaw):
    # tool z pointing down, tool x along yaw
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0],[s, -c, 0],[0, 0, -1.0]])
