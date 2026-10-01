import numpy as np, sys, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
import approach as A
from approach import fk_arm, LIM, Rpitch, Rz, HOME, qerr
from opt2_search import body_pts, cube, frames, OFF
PI = np.pi
LO, HI = -LIM.astype(float), LIM.astype(float)
for i in (0, 2, 4, 6): LO[i], HI[i] = HOME[i]-PI, HOME[i]+PI
HOV = A.HOVER
def pts(q):
    P = body_pts(q); return P
def clear(q):  # floor >0.03 for all body pts except fingertip pts; base box x<0.35 & z<0.42 forbidden
    P = pts(q)
    return np.concatenate([P[:-1, 2] - 0.03, [max(z - 0.42, x - 0.35) for x, _, z in P]])
def pose_err(q, pc, R):
    T = fk_arm(q); c = cube(T)
    return np.r_[c - pc, (T[:3, :3] - R).ravel()[[0, 2, 5, 8, 1]]]
QP = np.load('qp_nom.npy')
WT = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
def run(n, seed, free_yaw, gmax):
    rng = np.random.default_rng(seed); out = []
    def unpack(x): return x[:7], x[7:14], x[14], x[15], x[16], x[17], x[18]
    def obj(x): return x[18] + WT*(x[19] + x[20])
    def eqs(x):
        qh, qg, d, p, y, yaw, t = unpack(x)
        R = Rz(yaw) @ Rpitch(p); pc = np.array([d, y, 0.02])
        e = np.r_[pose_err(qg, pc, R), pose_err(qh, pc - HOV*R[:, 2], R)]
        return e
    def ineqs(x):
        qh, qg, d, p, y, yaw, t = unpack(x)
        tg, t2 = x[19], x[20]
        return np.r_[t - (qh-HOME), t + (qh-HOME), gmax - (qg-qh), gmax + (qg-qh), tg-(qg-qh), tg+(qg-qh), t2-(QP-qg), t2+(QP-qg), clear(qh), clear(qg)]
    ylim = 0.15 if free_yaw else 0.0; ywl = np.radians(40) if free_yaw else 0.0
    lb = np.r_[LO, LO, 0.45, np.radians(45), -ylim, -ywl, 0, 0, 0]; ub = np.r_[HI, HI, 0.9, np.radians(90), ylim, ywl, 4, 4, 4]
    for k in range(n):
        d, p = rng.uniform(0.45, 0.9), rng.uniform(np.radians(45), np.radians(90))
        y = rng.uniform(-ylim, ylim); yaw = rng.uniform(-ywl, ywl)
        R = Rz(yaw) @ Rpitch(p)
        u = rng.random(); s = HOME + rng.normal(0, 0.8, 7) if u < 0.4 else (QP + rng.normal(0, 0.4, 7) if u < 0.8 else rng.uniform(LO, HI))
        qg, e = A.ik_limited(np.array([d, y, 0.02]), R, s)
        if e > 5e-3: continue
        qh, e2 = A.ik_limited(np.array([d, y, 0.02]) - HOV*R[:, 2], R, qg)
        if e2 > 5e-3: continue
        # unwrap continuous joints near HOME
        for q in (qh, qg):
            for i in (0, 2, 4, 6): q[i] = HOME[i] + A.wrap(q[i]-HOME[i])
        x0 = np.r_[qh, qg, d, p, y, yaw, np.abs(qh-HOME).max(), np.abs(qg-qh).max(), np.abs(QP-qg).max()]
        r = minimize(obj, x0, method='SLSQP', bounds=list(zip(lb, ub)),
                     constraints=[{'type': 'eq', 'fun': eqs}, {'type': 'ineq', 'fun': ineqs}], options={'maxiter': 500, 'ftol': 1e-9})
        x = r.x
        if np.abs(eqs(x)).max() < 1e-4 and ineqs(x).min() > -1e-4:
            out.append((obj(x), x))
    out.sort(key=lambda a: a[0]); return out
if __name__ == '__main__':
    fy = sys.argv[1] == 'y'; gmax = float(sys.argv[2]); n = int(sys.argv[3]); sd = int(sys.argv[4])
    res = run(n, sd, fy, gmax)
    import pickle; pickle.dump(res, open('first_res_%s_%s_%d_w%s.pkl' % (sys.argv[1], sys.argv[2], sd, WT), 'wb'))
    for t, x in res[:3]:
        print('%.4f d=%.3f p=%.1f y=%.3f yaw=%.1f' % (t, x[14], np.degrees(x[15]), x[16], np.degrees(x[17])), np.round(x[:7], 3), 't1 %.3f tg %.3f t2 %.3f' % (x[18], x[19], x[20]))
