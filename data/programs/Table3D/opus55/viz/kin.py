import numpy as np
D = [-(0.1564+0.1284), -(0.0054+0.0064), -(0.2104+0.2104), -(0.0064+0.0064), -(0.2084+0.1059), 0.0, -(0.1059+0.0615)]
ALPHA = [np.pi/2]*6 + [np.pi]
OFF = [0, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
def dh(alpha, d, th):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(th), np.sin(th)
    return np.array([[ct, -st*ca, st*sa, 0],[st, ct*ca, -ct*sa, 0],[0, sa, ca, d],[0,0,0,1.]])
def fk(q, tool=0.0):
    T = np.diag([1.,-1.,-1.,1.])
    for i in range(7):
        T = T @ dh(ALPHA[i], D[i], q[i]+OFF[i])
    if tool:
        Tt = np.eye(4); Tt[2,3]=tool; T = T @ Tt
    return T
def rot_err(R, Rd):
    Re = Rd @ R.T
    return 0.5*np.array([Re[2,1]-Re[1,2], Re[0,2]-Re[2,0], Re[1,0]-Re[0,1]])
def ik(q0, p_des, R_des, tool=0.0, iters=200, wrot=0.5, tol=1e-4):
    q = np.array(q0, float)
    for it in range(iters):
        T = fk(q, tool)
        e = np.concatenate([p_des - T[:3,3], wrot*rot_err(T[:3,:3], R_des)])
        if np.linalg.norm(e) < tol: break
        J = np.zeros((6,7)); eps=1e-6
        for j in range(7):
            dq = q.copy(); dq[j]+=eps
            Tj = fk(dq, tool)
            J[:3,j] = (Tj[:3,3]-T[:3,3])/eps
            J[3:,j] = wrot*rot_err(T[:3,:3], Tj[:3,:3])/eps
        lam=1e-3
        dq = J.T @ np.linalg.solve(J@J.T + lam*np.eye(6), e)
        n=np.linalg.norm(dq)
        if n>0.3: dq*=0.3/n
        q += dq
    return q, np.linalg.norm(e)
def down_R(yaw):
    # gripper z axis pointing down, x axis at yaw
    c,s=np.cos(yaw),np.sin(yaw)
    return np.array([[c, s, 0],[s,-c,0],[0,0,-1.]])
