import numpy as np
# Kinova Gen3 7DOF classic DH (alpha, a, d, theta_offset)
DH = [(np.pi/2,0,-0.2848,0.0),(np.pi/2,0,-0.0118,np.pi),(np.pi/2,0,-0.4208,np.pi),
      (np.pi/2,0,-0.0128,np.pi),(np.pi/2,0,-0.3143,np.pi),(np.pi/2,0,0.0,np.pi),(np.pi,0,-0.1674,np.pi)]
def _T(alpha,a,d,th):
    ca,sa,ct,st=np.cos(alpha),np.sin(alpha),np.cos(th),np.sin(th)
    return np.array([[ct,-st*ca,st*sa,a*ct],[st,ct*ca,-ct*sa,a*st],[0,sa,ca,d],[0,0,0,1]])
T0 = np.diag([1.,-1.,-1.,1.])  # base -> frame 0 (alpha=pi)
def fk_arm(q, tool=0.0):
    T=T0.copy()
    for (al,a,d,off),qi in zip(DH,q):
        T=T@_T(al,a,d,qi+off)
    if tool: T=T@np.array([[1,0,0,0],[0,1,0,0],[0,0,1,tool],[0,0,0,1.]])
    return T

def jac_pos_rot(q, tool=0.0, eps=1e-5):
    T = fk_arm(q, tool)
    J = np.zeros((6, 7))
    for i in range(7):
        dq = np.array(q, float); dq[i] += eps
        T2 = fk_arm(dq, tool)
        J[:3, i] = (T2[:3, 3] - T[:3, 3]) / eps
        dR = T2[:3, :3] @ T[:3, :3].T
        J[3:, i] = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]]) / (2*eps)
    return T, J

def rot_err(R, Rd):
    E = Rd @ R.T
    return 0.5*np.array([E[2,1]-E[1,2], E[0,2]-E[2,0], E[1,0]-E[0,1]])

def ik(p_des, R_des, q0, tool=0.0, iters=100, rot_w=0.3, q_rest=None, rest_w=0.0):
    q = np.array(q0, float)
    for _ in range(iters):
        T, J = jac_pos_rot(q, tool)
        ep = p_des - T[:3, 3]
        er = rot_err(T[:3, :3], R_des) if R_des is not None else np.zeros(3)
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        e = np.concatenate([ep, rot_w*er])
        Jw = J.copy(); Jw[3:] *= rot_w
        if R_des is None: Jw[3:] = 0
        lam = 1e-3
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam*np.eye(6), e)
        if q_rest is not None and rest_w > 0:
            N = np.eye(7) - np.linalg.pinv(Jw) @ Jw
            dq += N @ (rest_w*(np.array(q_rest)-q))
        n = np.linalg.norm(dq)
        if n > 0.3: dq *= 0.3/n
        q += dq
    return q, np.linalg.norm(ep), np.linalg.norm(er)
