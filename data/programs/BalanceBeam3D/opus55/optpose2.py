import numpy as np
from scipy.optimize import minimize
from kin import Kin, _rz
np.set_printoptions(precision=4, suppress=True, linewidth=200)
k = Kin(mount=(0.113,0,0.36), tool=0.128)
qh = np.array([0, -0.349, 3.142, -2.548, 0, -0.873, 1.571])
def ry(a):
    c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def rot_err(R, Rt):
    dR = Rt @ R.T
    return 0.5*np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]])
BND=[(None,None),(-2.41,2.41),(None,None),(-2.66,2.66),(None,None),(-2.23,2.23),(None,None),(0.3,0.9),(-0.8,0.8)]
def solve(z, qref, maxtilt):
    def cons(v):
        q, rr, tilt = v[:7], v[7], v[8]
        p, R = k.fk((0,0,0), q)
        return np.concatenate([p - np.array([rr, 0, z]), rot_err(R, ry(tilt) @ _rz(np.pi/2))])
    def obj(v):
        d = np.abs(v[:7]-qref); m = d.max()
        return m + np.log(np.sum(np.exp(20*(d-m))))/20
    bnd = BND[:8] + [(-maxtilt, maxtilt)]
    best=None
    for trial in range(20):
        v0 = np.concatenate([qref + np.random.randn(7)*0.5*(trial>0), [0.5, 0.0]])
        res = minimize(obj, v0, constraints=[{"type":"eq","fun":cons}], method="SLSQP", bounds=bnd, options=dict(maxiter=500))
        if res.success and np.abs(cons(res.x)).max()<1e-4 and (best is None or res.fun<best.fun): best=res
    return best
np.random.seed(0)
for mt in [0.0, 0.3, 0.6]:
    b = solve(0.045, qh, mt)
    print(mt, "maxdev", np.abs(b.x[:7]-qh).max().round(3), "r", b.x[7].round(3), "tilt", b.x[8].round(3), b.x[:7])
