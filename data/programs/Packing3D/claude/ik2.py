import numpy as np
from fk import fk, JOINTS, TOOL, T, rpy, rotz
from ik import fk_full, jacobian

LIM_LO=np.array([-np.inf,-2.41,-np.inf,-2.66,-np.inf,-2.23,-np.inf])
LIM_HI=np.array([ np.inf, 2.41, np.inf, 2.66, np.inf, 2.23, np.inf])

def _err(q,tp,tR,tool_z):
    M,_=fk_full(q,tool_z)
    e=np.zeros(6); e[:3]=tp-M[:3,3]
    Rerr=tR@M[:3,:3].T
    w=np.array([Rerr[2,1]-Rerr[1,2],Rerr[0,2]-Rerr[2,0],Rerr[1,0]-Rerr[0,1]])
    s=np.linalg.norm(w); c=(np.trace(Rerr)-1)/2
    ang=np.arctan2(s/2,c)
    if s>1e-9: w=w/s*ang
    else: w=np.zeros(3)
    e[3:]=w
    return e,M

def solve_once(q0,tp,tR,tool_z=0.0,iters=200,rot_w=1.0):
    q=np.array(q0,float)
    for it in range(iters):
        e,M=_err(q,tp,tR,tool_z)
        ew=e.copy(); ew[3:]*=rot_w
        if np.linalg.norm(e[:3])<1e-4 and np.linalg.norm(e[3:])<1e-3: break
        J,_=jacobian(q,tool_z); J=J.copy(); J[3:]*=rot_w
        lam=0.08
        dq=J.T@np.linalg.solve(J@J.T+lam**2*np.eye(6),ew)
        n=np.linalg.norm(dq)
        if n>0.4: dq=dq/n*0.4
        q=q+dq
        q=np.clip(q,LIM_LO+1e-6,LIM_HI-1e-6)
    e,M=_err(q,tp,tR,tool_z)
    return q,M,np.linalg.norm(e[:3]),np.linalg.norm(e[3:])

def ik_solve(q0,tp,tR,tool_z=0.0,restarts=12,rng=None,pos_tol=2e-3,rot_tol=2e-2):
    rng=rng or np.random.default_rng(0)
    best=None
    cands=[np.array(q0,float)]
    for _ in range(restarts):
        cands.append(np.clip(np.array(q0,float)+rng.normal(0,1.2,7),LIM_LO+1e-6,LIM_HI-1e-6))
    for c in cands:
        q,M,pe,re=solve_once(c,tp,tR,tool_z)
        score=pe+0.05*re
        if best is None or score<best[0]:
            best=(score,q,M,pe,re)
        if pe<pos_tol and re<rot_tol:
            break
    return best[1],best[2],best[3],best[4]
