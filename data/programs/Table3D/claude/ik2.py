import numpy as np
from scipy.optimize import least_squares
from fk import fk

def residual(q, pos, tool_len, zdir):
    T = fk(q, tool_len)
    r = np.zeros(6)
    r[:3] = T[:3,3]-pos
    r[3:] = (T[:3,2]-zdir)*0.5
    return r

def solve_pos_axis(pos, tool_len=0.0, zdir=np.array([0,0,-1.0]), q0=None, n_restart=12, rng=None, tol=2e-3):
    rng = rng or np.random.default_rng(0)
    home=np.array([0,-0.35,-3.1416,-2.5,0,-0.87,1.5708])
    cands=[]
    if q0 is not None: cands.append(np.asarray(q0,float))
    cands.append(home)
    for _ in range(n_restart):
        cands.append(home+rng.normal(0,1.5,7))
    best=None
    for c in cands:
        res = least_squares(residual, c, args=(pos,tool_len,zdir), xtol=1e-10, ftol=1e-10, max_nfev=300)
        pe = np.linalg.norm(res.fun[:3]); ae=np.linalg.norm(res.fun[3:])
        sc = pe+ae
        if best is None or sc<best[0]: best=(sc,res.x,pe,ae)
        if sc<tol: break
    return best[1],best[2],best[3]
