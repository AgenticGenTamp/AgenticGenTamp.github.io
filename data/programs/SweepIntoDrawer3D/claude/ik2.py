import numpy as np
from fk import LINKS, TOOL, rotx, rotz, T
PI=np.pi
JLO=np.array([-3.1,-2.41,-3.1,-2.66,-3.1,-2.23,-3.1])
JHI=np.array([ 3.1, 2.41, 3.1, 2.66, 3.1, 2.23, 3.1])

def fk_all(q):
    M=np.eye(4); origins=[]; axes=[]
    for i,(xyz,rx) in enumerate(LINKS):
        M=M@T(rotx(rx),np.array(xyz))
        origins.append(M[:3,3].copy()); axes.append(M[:3,2].copy())
        M=M@T(rotz(q[i]),np.zeros(3))
    M=M@T(rotx(TOOL[1]),np.array(TOOL[0]))
    return np.array(origins), np.array(axes), M

def _solve(target_p,target_R,q0,iters=120,w_rot=0.6):
    q=np.array(q0,dtype=float)
    for k in range(iters):
        o,a,M=fk_all(q); p=M[:3,3]
        J=np.zeros((6,7))
        for i in range(7):
            J[:3,i]=np.cross(a[i],p-o[i]); J[3:,i]=a[i]
        ep=target_p-p
        Rerr=target_R@M[:3,:3].T
        ang=np.arccos(np.clip((np.trace(Rerr)-1)/2,-1,1))
        if ang<1e-9: er=np.zeros(3)
        else:
            axis=np.array([Rerr[2,1]-Rerr[1,2],Rerr[0,2]-Rerr[2,0],Rerr[1,0]-Rerr[0,1]])/(2*np.sin(ang))
            er=axis*ang
        e=np.concatenate([ep,w_rot*er])
        if np.linalg.norm(ep)<1e-5 and ang<1e-3: break
        lam=0.08
        dq=J.T@np.linalg.solve(J@J.T+lam**2*np.eye(6),e)
        # nullspace: push joints toward limit center
        mid=np.clip((JLO+JHI)/2,-1,1)
        gz=-(q-mid)*0.02
        N=np.eye(7)-J.T@np.linalg.solve(J@J.T+lam**2*np.eye(6),J)
        dq=dq+N@gz
        q=np.clip(q+np.clip(dq,-0.25,0.25),JLO,JHI)
    o,a,M=fk_all(q)
    Rerr=target_R@M[:3,:3].T
    ang=np.arccos(np.clip((np.trace(Rerr)-1)/2,-1,1))
    return q, np.linalg.norm(target_p-M[:3,3]), ang

def ik_multi(target_p,target_R,q_cur,n=12,seed=0,ptol=0.004,atol=0.08):
    rng=np.random.default_rng(seed)
    best=None; bestcost=1e18
    cands=[np.array(q_cur,dtype=float)]
    for i in range(n):
        cands.append(rng.uniform(np.maximum(JLO,-3.0),np.minimum(JHI,3.0)))
    for q0 in cands:
        q,ep,ea=_solve(np.asarray(target_p,dtype=float),target_R,q0)
        if ep<ptol and ea<atol:
            cost=np.abs(q-np.asarray(q_cur)).max()
            if cost<bestcost: bestcost=cost; best=q
    return best,bestcost
