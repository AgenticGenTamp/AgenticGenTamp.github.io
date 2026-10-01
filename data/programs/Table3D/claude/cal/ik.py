import numpy as np
from fk import fk

LIM_LO = np.array([-np.inf,-2.41,-np.inf,-2.66,-np.inf,-2.23,-np.inf])
LIM_HI = np.array([ np.inf, 2.41, np.inf, 2.66, np.inf, 2.23, np.inf])

def pose_err(T, Tt):
    pe = T[:3,3]-Tt[:3,3]
    Re = T[:3,:3] @ Tt[:3,:3].T
    # rotation vector
    c = (np.trace(Re)-1)/2
    c = np.clip(c,-1,1)
    ang = np.arccos(c)
    if ang < 1e-8:
        oe = np.zeros(3)
    else:
        oe = ang/(2*np.sin(ang))*np.array([Re[2,1]-Re[1,2],Re[0,2]-Re[2,0],Re[1,0]-Re[0,1]])
    return np.concatenate([pe,oe])

def jac(q, tool_len, eps=1e-5):
    J = np.zeros((6,7))
    T0 = fk(q,tool_len)
    for i in range(7):
        dq = q.copy(); dq[i]+=eps
        T1 = fk(dq,tool_len)
        J[:,i] = pose_err(T1,T0)/eps
    return J

def ik(Tt, q0, tool_len=0.0, iters=200, tol=1e-4, pos_only=False):
    q = np.array(q0,dtype=float)
    for _ in range(iters):
        T = fk(q,tool_len)
        e = pose_err(T,Tt)
        if pos_only: e[3:]=0
        if np.linalg.norm(e) < tol:
            break
        J = jac(q,tool_len)
        if pos_only: J = J[:3]; ee=e[:3]
        else: ee=e
        lam=0.05
        dq = J.T @ np.linalg.solve(J@J.T + lam**2*np.eye(J.shape[0]), -ee)
        dq = np.clip(dq,-0.3,0.3)
        q = q+dq
        q = np.clip(q,LIM_LO,LIM_HI)
    return q, np.linalg.norm(pose_err(fk(q,tool_len),Tt)[:3]), np.linalg.norm(pose_err(fk(q,tool_len),Tt)[3:])

def solve(Tt, tool_len=0.0, seeds=None, n_rand=30, rng=None):
    rng = rng or np.random.default_rng(0)
    best=None
    cands=[]
    if seeds is not None: cands += list(seeds)
    home=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
    cands.append(home)
    for _ in range(n_rand):
        cands.append(np.clip(home + rng.normal(0,1.2,7), np.maximum(LIM_LO,-3.14), np.minimum(LIM_HI,3.14)))
    for q0 in cands:
        q,pe,oe = ik(Tt,q0,tool_len)
        sc = pe+0.1*oe
        if best is None or sc<best[0]:
            best=(sc,q,pe,oe)
        if pe<1e-3 and oe<1e-2: break
    return best[1],best[2],best[3]
