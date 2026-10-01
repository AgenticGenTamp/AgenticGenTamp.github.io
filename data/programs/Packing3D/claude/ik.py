import numpy as np
from fk import fk, JOINTS, TOOL, T, rpy, rotz

def fk_full(q, tool_z=0.0):
    Ms=[]; M=np.eye(4)
    for i,(xyz,r) in enumerate(JOINTS):
        M = M @ T(np.array(xyz), rpy(*r)) @ T(np.zeros(3), rotz(q[i]))
        Ms.append(M.copy())
    M = M @ T(np.array(TOOL[0]), rpy(*TOOL[1])) @ T(np.array([0,0,tool_z]), np.eye(3))
    return M, Ms

def jacobian(q, tool_z=0.0):
    M, Ms = fk_full(q, tool_z)
    p_e = M[:3,3]
    J = np.zeros((6,7))
    for i in range(7):
        z = Ms[i][:3,2]; p = Ms[i][:3,3]
        J[:3,i] = np.cross(z, p_e-p)
        J[3:,i] = z
    return J, M

def ik(q0, target_pos, target_R=None, tool_z=0.0, iters=300, pos_w=1.0, rot_w=0.5):
    q = np.array(q0, dtype=float)
    for it in range(iters):
        J, M = jacobian(q, tool_z)
        e = np.zeros(6)
        e[:3] = (target_pos - M[:3,3])*pos_w
        if target_R is not None:
            Rerr = target_R @ M[:3,:3].T
            # axis-angle
            w = np.array([Rerr[2,1]-Rerr[1,2], Rerr[0,2]-Rerr[2,0], Rerr[1,0]-Rerr[0,1]])
            s = np.linalg.norm(w); c = (np.trace(Rerr)-1)/2
            ang = np.arctan2(s/2, c)
            if s > 1e-8: w = w/s*ang
            else: w = np.zeros(3)
            e[3:] = w*rot_w
        else:
            J = J[:3]; e = e[:3]
        lam = 0.05
        dq = J.T @ np.linalg.solve(J@J.T + lam**2*np.eye(len(e)), e)
        n = np.linalg.norm(dq)
        if n>0.3: dq = dq/n*0.3
        q = q + dq
        if np.linalg.norm(e[:3])<1e-5 and (target_R is None or np.linalg.norm(e[3:])<1e-4):
            break
    J,M = jacobian(q, tool_z)
    return q, M
