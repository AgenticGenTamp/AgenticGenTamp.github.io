import numpy as np
import fk

def pose_error(Tcur, p_des, R_des):
    ep = p_des - Tcur[:3, 3]
    Rerr = R_des @ Tcur[:3, :3].T
    # log of rotation
    c = (np.trace(Rerr) - 1) / 2
    c = np.clip(c, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-8:
        w = np.zeros(3)
    else:
        w = ang / (2 * np.sin(ang)) * np.array([Rerr[2,1]-Rerr[1,2], Rerr[0,2]-Rerr[2,0], Rerr[1,0]-Rerr[0,1]])
    return ep, w

def ik_step(q, p_des, R_des, tool_z=0.0, damp=0.05, w_rot=1.0, max_delta=0.2):
    T = fk.fk_ee(q, tool_z)
    ep, ew = pose_error(T, p_des, R_des)
    e = np.concatenate([ep, w_rot * ew])
    J = fk.jacobian(q, tool_z)
    J = np.vstack([J[:3], w_rot * J[3:]])
    JT = J.T
    dq = JT @ np.linalg.solve(J @ JT + damp**2 * np.eye(6), e)
    n = np.max(np.abs(dq))
    if n > max_delta:
        dq = dq * (max_delta / n)
    return dq, np.linalg.norm(ep), np.linalg.norm(ew)

def solve_ik(q0, p_des, R_des, tool_z=0.0, iters=300, tol=1e-4):
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        dq, ep, ew = ik_step(q, p_des, R_des, tool_z, damp=0.02, max_delta=0.3)
        q = q + dq
        if ep < tol and ew < 1e-3:
            break
    return q, ep, ew
