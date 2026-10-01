import numpy as np
from scipy.optimize import minimize
from kin import Kin, _rz
np.set_printoptions(precision=4, suppress=True, linewidth=200)
k = Kin(mount=(0.113,0,0.36), tool=0.128)
qh = np.array([0, -0.349, 3.142, -2.548, 0, -0.873, 1.571])
def rot_err(R, Rt):
    dR = Rt @ R.T
    return 0.5*np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]])
def solve(z, qref, r=None, q0=None, smooth=20):
    # vars: q(7), r
    def cons(v):
        q, rr = v[:7], v[7]
        p, R = k.fk((0,0,0), q)
        return np.concatenate([p - np.array([rr, 0, z]), rot_err(R, _rz(np.pi/2))])
    def obj(v):
        d = v[:7]-qref
        return np.log(np.sum(np.exp(smooth*np.abs(d)-smooth*np.abs(d).max())))/smooth+np.abs(d).max()
    cs = [{"type":"eq","fun":cons}]
    if r is not None: cs.append({"type":"eq","fun":lambda v: v[7]-r})
    best=None
    for trial in range(10):
        v0 = np.concatenate([(q0 if q0 is not None else qref) + np.random.randn(7)*0.3*(trial>0), [r if r else 0.5]])
        res = minimize(obj, v0, constraints=cs, method="SLSQP", bounds=BND, options=dict(maxiter=500))
        if res.success and np.abs(cons(res.x)).max()<1e-4 and (best is None or res.fun<best.fun): best=res
    return best
BND=[(None,None),(-2.41,2.41),(None,None),(-2.66,2.66),(None,None),(-2.23,2.23),(None,None),(0.35,0.9)]
np.random.seed(0)
C = solve(0.08, qh)
print("carry", C.x, "maxdev", np.abs(C.x[:7]-qh).max())
G = solve(0.005, C.x[:7], r=C.x[7])
print("grasp", G.x, "maxdev from C", np.abs(G.x[:7]-C.x[:7]).max())
for r in [0.4,0.5,0.6,0.7]:
    C2 = solve(0.08, qh, r=r); G2 = solve(0.005, C2.x[:7], r=r)
    print(r, "home->C", np.abs(C2.x[:7]-qh).max(), "C->G", np.abs(G2.x[:7]-C2.x[:7]).max())
print("=====")
for r in [0.45, 0.5, 0.55]:
    C2 = solve(0.08, qh, r=r); P2 = solve(0.045, C2.x[:7], r=r); G2 = solve(0.005, P2.x[:7], r=r)
    print(r, "C", list(C2.x[:7].round(4)), "\n  P", list(P2.x[:7].round(4)), "\n  G", list(G2.x[:7].round(4)), np.abs(C2.x[:7]-qh).max(), np.abs(G2.x[:7]-C2.x[:7]).max())
