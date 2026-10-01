import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
sys.argv = ['x', 'n', '0.2', '0', '0', '1.0']
import first_search as F
from scipy.optimize import minimize
from approach import *
qp = F.QP
import approach as A
def _pe(q, pc, Rd):
    T = fk_arm(q); return np.r_[F.cube(T) - pc, A._rot_err(T[:3, :3], Rd)]
F.pose_err = _pe
R = Rpitch(PICK_PITCH)
qh0, _ = solve(np.array([PICK_D, 0, 0.02]) - HOVER*R[:, 2], R, [Q_PICK0]); qg0, _ = solve(np.array([PICK_D, 0, 0.02]), R, [qh0])
def mk(w1, fy):
    def obj(x): return w1*x[18] + x[19] + x[20]
    def eqs(x):
        qh, qg = x[:7], x[7:14]; d, p, y, yaw = x[14:18]
        Rr = Rz(yaw) @ Rpitch(p); pc = np.array([d, y, 0.02])
        return np.r_[F.pose_err(qg, pc, Rr), F.pose_err(qh, pc - HOVER*Rr[:, 2], Rr), ([] if fy else [y, yaw])]
    def ineqs(x):
        qh, qg = x[:7], x[7:14]; t, tg, t2 = x[18:21]
        return np.r_[t-(qh-HOME), t+(qh-HOME), tg-(qg-qh), tg+(qg-qh), t2-(qp-qg), t2+(qp-qg), F.clear(qh), F.clear(qg)]
    return obj, eqs, ineqs
lb = np.r_[F.LO, F.LO, 0.45, np.radians(45), -0.15, -0.7, 0, 0, 0]; ub = np.r_[F.HI, F.HI, 0.9, np.radians(90), 0.15, 0.7, 4, 4, 4]
rng = np.random.default_rng(0)
for fy in (False, True):
  for w1 in (1.0, 1.5, 2.0, 3.0):
    obj, eqs, ineqs = mk(w1, fy); best = None
    for k in range(8):
        x0 = np.r_[qh0, qg0, PICK_D, PICK_PITCH, 0, 0, 0, 0, 0] + (0 if k == 0 else np.r_[rng.normal(0, 0.1, 14), rng.normal(0, 0.05, 2), 0, 0, 0, 0, 0])
        x0[18] = np.abs(x0[:7]-HOME).max(); x0[19] = np.abs(x0[7:14]-x0[:7]).max(); x0[20] = np.abs(qp-x0[7:14]).max()
        r = minimize(obj, x0, method='SLSQP', bounds=list(zip(lb, ub)), constraints=[{'type': 'eq', 'fun': eqs}, {'type': 'ineq', 'fun': ineqs}], options={'maxiter': 600, 'ftol': 1e-10})
        x = r.x
        if np.abs(eqs(x)).max() < 1e-4 and ineqs(x).min() > -1e-4 and (best is None or obj(x) < obj(best)): best = x
    x = best
    print('yaw' if fy else 'noyaw', 'w1=%.1f d=%.3f p=%.2f y=%.3f yaw=%.1f t1 %.3f tg %.3f t2 %.3f sum %.3f' % (w1, x[14], np.degrees(x[15]), x[16], np.degrees(x[17]), x[18], x[19], x[20], x[18:21].sum()), list(np.round(x[:7], 4)))
print('current t1 %.3f tg %.3f t2 %.3f' % (np.abs(qh0-HOME).max(), np.abs(qg0-qh0).max(), np.abs(qp-qg0).max()))
