import numpy as np, sys, warnings, pickle
CASE = sys.argv[1] + '_' + sys.argv[2]
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from approach import fk_arm, MX, H, L, LIM, Rpitch, solve, cube_pos_from_q, _DH, _T0, _T
OFF = np.array([MX, 0, H])
def frames(q):
    T = _T0.copy(); out = []
    for (al, a, d, off), qi in zip(_DH, q):
        T = T @ _T(al, a, d, qi + off); out.append(T.copy())
    return out
def cube(T): return T[:3, 3] + OFF + L*T[:3, 2]
def body_pts(q):
    Fs = frames(q); pts = [F[:3, 3] + OFF for F in Fs[1:]]
    T = Fs[-1]; p = T[:3, 3] + OFF
    for s in (0.0, 0.06, 0.11):
        pts.append(p + s*T[:3, 2])
    return np.array(pts)
MARG = 0.06
def cab_margins(q, d2, zc):
    P = body_pts(q); front = d2 - 0.15
    ceil = [max(zc + 0.23 - MARG - z, front - x) for x, _, z in P]
    shelf = [max(z - (zc - 0.02 + 0.03), front - x) for x, _, z in P[:-1]]  # skip near-finger point
    return np.array(ceil + shelf)
def tip_z(q):
    T = fk_arm(q); return T[2, 3] + H + (L + 0.02)*T[2, 2]
def unpack(x): return x[:7], x[7:14], x[14], x[15], x[16], x[17], x[18], x[19:26]
ZB = 0.69
def run(zc=0.61, p1fix=None, place_x_horiz=True, n=60, seed=0):
    rng = np.random.default_rng(seed); res = []
    def eqs(x):
        q1, q2, d1, p1, d2, p2, t, q3 = unpack(x)
        T1, T2 = fk_arm(q1), fk_arm(q2)
        c1, c2 = cube(T1), cube(T2)
        e = [c1[0]-d1, c1[1], c1[2]-0.02, T1[1, 2], T1[2, 2]+np.sin(p1), T1[0, 0], T1[2, 0],
             c2[0]-d2, c2[1], c2[2]-zc, T2[1, 2], T2[2, 2]+np.sin(p2)]
        if place_x_horiz: e.append(T2[2, 0])
        T3 = fk_arm(q3); c3 = cube(T3)
        e += [c3[0]-d2, c3[1], c3[2]-ZB] + list((T3[:3, :3] - T2[:3, :3]).ravel()[[2, 5, 0]])
        return np.array(e)
    def ineqs(x):
        q1, q2, d1, p1, d2, p2, t, q3 = unpack(x); dq = q2 - q1
        T1, T2 = fk_arm(q1), fk_arm(q2)
        dq3 = q3 - q1
        return np.concatenate([t - dq, t + dq, t - dq3, t + dq3, [T1[0, 2], T2[0, 2]], cab_margins(q2, d2, zc), cab_margins(q3, d2, ZB)])
    lb = np.r_[-LIM, -LIM, 0.4, np.radians(45), 0.6, np.radians(-10), 0, -LIM]
    ub = np.r_[LIM, LIM, 0.8, np.radians(90), 0.95, np.radians(30), 5, LIM]
    if p1fix is not None: lb[15] = ub[15] = np.radians(p1fix)
    X0 = pickle.load(open('opt2_x0_%s.pkl' % CASE, 'rb'))
    for k in range(len(X0)):
        x0 = X0[k]
        r = minimize(lambda x: x[18], x0, method='SLSQP', bounds=list(zip(lb, ub)),
                     constraints=[{'type': 'eq', 'fun': eqs}, {'type': 'ineq', 'fun': ineqs}],
                     options={'maxiter': 400, 'ftol': 1e-8})
        x = r.x
        if np.abs(eqs(x)).max() < 1e-4 and ineqs(x).min() > -1e-4:
            res.append((x[18], x))
    res.sort(key=lambda a: a[0])
    return res
if __name__ == '__main__':
    import pickle
    p1fix = None if sys.argv[1] == 'free' else float(sys.argv[1])
    horiz = sys.argv[2] == 'h'
    res = run(p1fix=p1fix, place_x_horiz=horiz, n=int(sys.argv[3]), seed=int(sys.argv[4]))
    pickle.dump(res, open('opt2_res_%s_%s_ws%s.pkl' % (sys.argv[1], sys.argv[2], sys.argv[4]), 'wb'))
    for t, x in res[:5]:
        print(round(t, 4), 'p1=%.1f d1=%.3f d2=%.3f p2=%.1f' % (np.degrees(x[15]), x[14], x[16], np.degrees(x[17])))
