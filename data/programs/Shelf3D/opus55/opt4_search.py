import numpy as np, sys, warnings, pickle
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from approach import fk_arm, LIM, Rpitch
from opt2_search import cab_margins, cube, frames, OFF
from opt3_search import ik_b, unpack, ZA, ZB
DBG = 0
PI = np.pi
def branch_bounds(v):
    lo, hi = -LIM.astype(float).copy(), LIM.astype(float).copy()
    w = (0.6, 0.8, 0.8) if v == 'R' else (1.0, 1.2, 1.2)
    lo[0], hi[0] = -w[0], w[0]; lo[2], hi[2] = PI-w[1], PI+w[1]; lo[4], hi[4] = -w[2], w[2]
    return lo, hi
def links(q): return np.array([F[:3, 3] + OFF for F in frames(q)[:-1]])  # intermediate origins
def pick_link_m(q): return links(q)[:, 2] - 0.10
def place_link_m(q, d2, zc):
    P = links(q); fr = d2 - 0.2
    return np.array([max(z - (zc + 0.01), fr - x) for x, _, z in P] + [max(zc + 0.21 - z, fr - x) for x, _, z in P])
D1R, P1R, D2R, P2R = (0.35, 0.9), np.radians([45, 90]), (0.6, 1.15), np.radians([-15, 35])
def run(v, horiz, n, seed):
    rng = np.random.default_rng(seed); res = []
    qlo, qhi = branch_bounds(v)
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
                               cab_margins(q2, d2, ZA), cab_margins(q3, d2, ZB),
                               pick_link_m(q1), pick_link_m(qh), place_link_m(q2, d2, ZA), place_link_m(q3, d2, ZB)])
    lb = np.r_[qlo, qlo, D1R[0], P1R[0], D2R[0], P2R[0], 0, qlo, qlo]
    ub = np.r_[qhi, qhi, D1R[1], P1R[1], D2R[1], P2R[1], 5, qhi, qhi]
    for k in range(n):
        d1, p1, d2, p2 = rng.uniform(*D1R), rng.uniform(*P1R), rng.uniform(*D2R), rng.uniform(*P2R)
        s = rng.uniform(qlo, qhi); s[[0, 2, 4, 6]] = rng.uniform(np.maximum(qlo, -PI), np.minimum(qhi, PI))[[0, 2, 4, 6]]
        q1, e1 = ik_b(np.array([d1, 0, 0.02]), Rpitch(p1), s, qlo, qhi)
        R2 = Rpitch(p2) @ (np.diag([-1., -1, 1]) if rng.random() < 0.5 else np.eye(3))
        if not horiz:
            r = rng.uniform(-0.6, 0.6); c, sn = np.cos(r), np.sin(r); R2 = R2 @ np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        q2, e2 = ik_b(np.array([d2, 0, ZA]), R2, q1, qlo, qhi)
        q3, e3 = ik_b(np.array([d2, 0, ZB]), fk_arm(q2)[:3, :3], q2, qlo, qhi)
        T1 = fk_arm(q1); qh, eh = ik_b(cube(T1) - 0.05*T1[:3, 2], T1[:3, :3], q1, qlo, qhi)
        if max(e1, e2, e3, eh) > 5e-3: continue
        x0 = np.r_[q1, q2, d1, p1, d2, p2, 1.0, q3, qh]
        x0[18] = max(np.abs(q2-q1).max(), np.abs(q3-q1).max())
        r = minimize(lambda x: x[18], x0, method='SLSQP', bounds=list(zip(lb, ub)),
                     constraints=[{'type': 'eq', 'fun': eqs}, {'type': 'ineq', 'fun': ineqs}],
                     options={'maxiter': 400, 'ftol': 1e-8})
        x = r.x
        if np.abs(eqs(x)).max() < 1e-4 and ineqs(x).min() > -1e-4:
            res.append((x[18], x))
    res.sort(key=lambda a: a[0])
    return res
if __name__ == '__main__':
    v, h, n, sd = sys.argv[1], sys.argv[2] == 'h', int(sys.argv[3]), int(sys.argv[4])
    res = run(v, h, n, sd)
    pickle.dump(res, open('opt4_res_%s_%s_%d.pkl' % (v, sys.argv[2], sd), 'wb'))
    for t, x in res[:3]:
        print(round(t, 4), 'p1=%.1f d1=%.3f d2=%.3f p2=%.1f' % (np.degrees(x[15]), x[14], x[16], np.degrees(x[17])))
