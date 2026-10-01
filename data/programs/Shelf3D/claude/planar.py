import numpy as np
import armkin as ak

LAT = None  # lateral offset, computed below

def qfull(q2,q4,q6):
    return np.array([0.0,q2,0.0,q4,0.0,q6,0.0])

def _p(q2,q4,q6):
    T=ak.fk(qfull(q2,q4,q6),ak.TOOL_OFFSET)
    return T[:3,3]

LAT = float(_p(0.0,0.0,0.0)[1])

def planar_ik(xp, zp, s, seed=None, iters=80, cur=None):
    """Solve q2,q4,q6 with q2+q4+q6=s so that tool is at (xp, LAT, zp)."""
    best=None
    seeds=[] if seed is None else [seed]
    seeds += [(0.6,-1.2),(1.2,-1.8),(0.3,-0.6),(1.6,-2.2),(-0.6,1.2),(0.9,-2.4),(1.9,-1.0),(0.2,-2.0),(1.4,-0.4),
              (0.5,-2.5),(2.0,-2.0),(1.0,-1.0),(-1.0,2.0),(0.0,-1.5),(2.2,-0.8),(1.7,-2.5),(0.8,-0.2),(-0.3,-1.0),
              (1.1,-2.9),(0.4,-1.9),(2.1,-1.6),(1.3,-1.3),(0.7,-2.7),(1.5,-0.9)]
    for s0 in seeds:
        a,b=float(s0[0]),float(s0[1])
        ok=False
        for it in range(iters):
            c=s-a-b
            p=_p(a,b,c)
            e=np.array([xp-p[0], zp-p[2]])
            if np.linalg.norm(e)<1e-6:
                ok=True; break
            J=np.zeros((2,2)); h=1e-5
            for k,(da,db) in enumerate([(h,0),(0,h)]):
                pp=_p(a+da,b+db,s-(a+da)-(b+db))
                J[:,k]=(np.array([pp[0],pp[2]])-np.array([p[0],p[2]]))/h
            try:
                d=np.linalg.solve(J+1e-9*np.eye(2), e)
            except np.linalg.LinAlgError:
                break
            n=np.linalg.norm(d)
            if n>0.4: d=d*(0.4/n)
            a+=d[0]; b+=d[1]
        if not ok: continue
        c=s-a-b
        a=ak.wrap(a); b=ak.wrap(b); c=ak.wrap(c)
        if abs(a)>2.24 or abs(b)>2.57 or abs(c)>2.09: continue
        if cur is None:
            cost=abs(a)+abs(b)+abs(c)
        else:
            cost=(abs(ak.wrap(a-cur[0]))+abs(ak.wrap(b-cur[1]))+abs(ak.wrap(c-cur[2])))*10.0+0.01*(abs(a)+abs(b)+abs(c))
        if best is None or cost<best[0]: best=(cost,(a,b,c))
    if best is None: return None
    return qfull(*best[1])
