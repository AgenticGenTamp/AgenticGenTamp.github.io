import numpy as np, sys, warnings, pickle
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from approach import fk_arm, LIM, Rpitch, solve
from opt2_search import cab_margins, cube, body_pts
ZA, ZB = 0.59, 0.69
DBG = 0
PI = np.pi
def branch_bounds(branch):
    lo, hi = -LIM.astype(float).copy(), LIM.astype(float).copy()
    if branch == 'A':
        lo[0], hi[0] = -0.6, 0.6; lo[2], hi[2] = PI-0.8, PI+0.8; lo[4], hi[4] = -0.8, 0.8
    return lo, hi
from approach import _jac, _rot_err, MX, H, L
def ik_b(p_des, R_des, q0, lo, hi, iters=300, rot_w=0.5):
    q = np.clip(np.array(q0, float), lo, hi)
    f = np.array(p_des) - np.array([MX, 0, H]) - L*R_des[:, 2]
    for _ in range(iters):
        T, J = _jac(q)
        ep = f - T[:3, 3]; er = _rot_err(T[:3, :3], R_des)
        if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(er) < 2e-3: break
        e = np.concatenate([ep, rot_w*er]); Jw = J.copy(); Jw[3:] *= rot_w
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-3*np.eye(6), e)
        nn = np.linalg.norm(dq)
        if nn > 0.2: dq *= 0.2/nn
        q = np.clip(q + dq, lo, hi)
    return q, float(np.linalg.norm(ep) + 0.2*np.linalg.norm(er))
def unpack(x): return x[:7], x[7:14], x[14], x[15], x[16], x[17], x[18], x[19:26], x[26:33]
def run(branch, horiz, n, seed):
    rng = np.random.default_rng(seed); res = []
    qlo, qhi = branch_bounds(branch)
    def eqs(x):
        q1, q2, d1, p1, d2, p2, t, q3, qh = unpack(x)
        T1, T2, T3, Th = fk_arm(q1), fk_arm(q2), fk_arm(q3), fk_arm(qh)
        c1, c2, c3, ch = cube(T1), cube(T2), cube(T3), cube(Th)
        e = [c1[0]-d1, c1[1], c1[2]-0.02, T1[1, 2], T1[2, 2]+np.sin(p1), T1[0, 0], T1[2, 0],
             c2[0]-d2, c2[1], c2[2]-ZA, T2[1, 2], T2[2, 2]+np.sin(p2)]
        if horiz: e.append(T2[2, 0])
        e += [c3[0]-d2, c3[1], c3[2]-ZB] + list((T3[:3, :3] - T2[:3, :3]).ravel()[[2, 5, 0]])
        e += list(ch - (c1 - 0.05*T1[:3, 2])) + list((Th[:3, :3] - T1[:3, :3]).ravel()[[2, 5, 0]])
        return np.array(e)
    def ineqs(x):
        q1, q2, d1, p1, d2, p2, t, q3, qh = unpack(x)
        T1, T2 = fk_arm(q1), fk_arm(q2)
        return np.concatenate([t-(q2-q1), t+(q2-q1), t-(q3-q1), t+(q3-q1), [T1[0, 2], T2[0, 2]],
                               cab_margins(q2, d2, ZA), cab_margins(q3, d2, ZB)])
    lb = np.r_[qlo, qlo, 0.4, np.radians(45), 0.6, np.radians(-10), 0, qlo, qlo]
    ub = np.r_[qhi, qhi, 0.8, np.radians(90), 0.95, np.radians(30), 5, qhi, qhi]
    for k in range(n):
        d1, p1, d2, p2 = rng.uniform(.4, .8), rng.uniform(lb[15], ub[15]), rng.uniform(.6, .95), rng.uniform(-0.17, 0.5)
        s = rng.uniform(qlo, qhi); s[[0, 2, 4, 6]] = rng.uniform(np.maximum(qlo, -PI), np.minimum(qhi, PI))[[0, 2, 4, 6]]
        if branch == 'A': s[2] = rng.uniform(PI-0.8, PI+0.8)
        q1, e1 = ik_b(np.array([d1, 0, 0.02]), Rpitch(p1), s, qlo, qhi)
        R2 = Rpitch(p2) @ (np.diag([-1., -1, 1]) if rng.random() < 0.5 else np.eye(3))
        if not horiz:
            r = rng.uniform(-0.6, 0.6); c, sn = np.cos(r), np.sin(r); R2 = R2 @ np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        q2, e2 = ik_b(np.array([d2, 0, ZA]), R2, q1, qlo, qhi)
        q3, e3 = ik_b(np.array([d2, 0, ZB]), fk_arm(q2)[:3, :3], q2, qlo, qhi)
        T1 = fk_arm(q1); qh, eh = ik_b(cube(T1) - 0.05*T1[:3, 2], T1[:3, :3], q1, qlo, qhi)
        if max(e1, e2, e3, eh) > 5e-3:
            if DBG: print(k, "init fail", e1, e2, e3, eh)
            continue
        x0 = np.r_[q1, q2, d1, p1, d2, p2, 1.0, q3, qh]
        x0[18] = max(np.abs(q2-q1).max(), np.abs(q3-q1).max())
        r = minimize(lambda x: x[18], x0, method='SLSQP', bounds=list(zip(lb, ub)),
                     constraints=[{'type': 'eq', 'fun': eqs}, {'type': 'ineq', 'fun': ineqs}],
                     options={'maxiter': 400, 'ftol': 1e-8})
        x = r.x
        if DBG: print(k, max(e1,e2,e3,eh), r.message, np.abs(eqs(x)).max(), ineqs(x).min(), x[18])
        if np.abs(eqs(x)).max() < 1e-4 and ineqs(x).min() > -1e-4:
            res.append((x[18], x))
    res.sort(key=lambda a: a[0])
    return res
if __name__ == '__main__':
    b, h, n, sd = sys.argv[1], sys.argv[2] == 'h', int(sys.argv[3]), int(sys.argv[4])
    res = run(b, h, n, sd)
    pickle.dump(res, open('opt3_res_%s_%s_%d.pkl' % (b, sys.argv[2], sd), 'wb'))
    for t, x in res[:3]:
        print(round(t, 4), 'p1=%.1f d1=%.3f d2=%.3f p2=%.1f' % (np.degrees(x[15]), x[14], x[16], np.degrees(x[17])))
