import numpy as np, sys, pickle, glob, warnings
warnings.filterwarnings('ignore')
from approach import fk_arm, LIM, solve, H, L
from opt2_search import cab_margins, cube, body_pts
def tipz_path(q1, q2):
    m = 1e9
    for s in np.linspace(0, 1, 101):
        q = q1 + s*(q2 - q1); T = fk_arm(q)
        m = min(m, T[2, 3] + H + (L + 0.02)*T[2, 2], body_pts(q)[:, 2].min())
    return m
def validate(x):
    q1, q2, d1, p1, d2, p2, t = x[:7], x[7:14], x[14], x[15], x[16], x[17], x[18]
    T1 = fk_arm(q1); R1 = T1[:3, :3]; T2 = fk_arm(q2); R2 = T2[:3, :3]
    qh, eh = solve(cube(T1) - 0.05*R1[:, 2], R1, [q1])
    out = dict(hover_err=eh, hover_dq=np.abs(qh - q1).max(), floor_min=tipz_path(q1, q2))
    worst = 0; ok = True; qs = q2
    for zc in np.linspace(0.61, 0.69, 9):
        sd = q2 + (zc - 0.61)/0.08*(x[19:26] - q2) if len(x) > 19 else qs
        q, e = solve(np.array([d2, 0, zc]), R2, [sd]); qs = q
        if abs(zc - 0.65) < 1e-6: qn = q
        m = cab_margins(q, d2, zc).min()
        worst = max(worst, np.abs(q - q1).max())
        ok &= (e < 2e-3) and m > -1e-3 and np.all(np.abs(q) <= LIM + 1e-6) and tipz_path(q1, q) > 0
    out.update(qn=qn, dqn=np.abs(qn - q1).max(), zc_ok=ok, worst_dq=worst, roll_x=R2[:, 0].round(2))
    return out
if __name__ == '__main__':
    for case in ['free_h', 'free_a', '90_h', '90_a']:
        res = []
        for f in glob.glob('opt2_res_%s_*.pkl' % case):
            res += pickle.load(open(f, 'rb'))
        res.sort(key=lambda a: a[0])
        print('==', case, len(res))
        shown = 0; last = -1
        for t, x in res:
            if abs(t - last) < 1e-3: continue
            v = validate(x); last = t
            print(round(t, 4), 'p1=%.1f d1=%.3f d2=%.3f p2=%.1f' % (np.degrees(x[15]), x[14], x[16], np.degrees(x[17])), {k: (np.round(w, 4) if k not in ('qn', 'zc_ok') else (w if k == 'zc_ok' else '')) for k, w in v.items()})
            if v['zc_ok'] and v['hover_err'] < 2e-3 and v['floor_min'] > 0:
                print('  q1', x[:7].round(3).tolist(), '\n  q2(0.65)', v['qn'].round(3).tolist())
                shown += 1
            if shown >= 2: break
