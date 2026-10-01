import numpy as np, pickle, glob, json, warnings, sys
warnings.filterwarnings('ignore')
from approach import fk_arm, LIM, solve, H, L
from opt2_search import cab_margins, cube, body_pts
from opt3_search import branch_bounds, ZA, ZB
def minz_path(qa, qb):
    m = 1e9
    for s in np.linspace(0, 1, 51):
        q = qa + s*(qb - qa); T = fk_arm(q)
        m = min(m, T[2, 3] + H + (L + 0.02)*T[2, 2], body_pts(q)[:, 2].min())
    return m
def validate(x, branch):
    lo, hi = branch_bounds(branch)
    q1, qa, d1, p1, d2, p2, t, qb, qh = x[:7], x[7:14], x[14], x[15], x[16], x[17], x[18], x[19:26], x[26:33]
    R2 = fk_arm(qa)[:3, :3]
    ok = np.all(q1 >= lo-1e-6) and np.all(q1 <= hi+1e-6) and np.all(qh >= lo-1e-6) and np.all(qh <= hi+1e-6)
    fl = min(minz_path(q1, qh), 1.0)
    worst = 0
    for zc in np.linspace(ZA, ZB, 11):
        sd = qa + (zc - ZA)/(ZB - ZA)*(qb - qa)
        q, e = solve(np.array([d2, 0, zc]), R2, [sd])
        m = cab_margins(q, d2, zc).min()
        worst = max(worst, np.abs(q - q1).max())
        fl = min(fl, minz_path(q1, q), minz_path(qh, q))
        ok &= bool(e < 2e-3 and m > -1e-3 and np.all(q >= lo-1e-6) and np.all(q <= hi+1e-6))
    ok &= fl > 0.0
    return bool(ok), worst, fl
def roll_deg(R):  # rotation of tool x about tool z away from horizontal
    return float(np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1))))
if __name__ == '__main__':
    out = {}
    for b in ['A', 'B']:
        res = []
        for f in glob.glob('opt3_res_%s_*.pkl' % b): res += pickle.load(open(f, 'rb'))
        res.sort(key=lambda a: a[0]); print('==', b, len(res))
        last = -1
        for t, x in res:
            if abs(t - last) < 1e-4: continue
            last = t; ok, worst, fl = validate(x, b)
            print(round(t, 4), ok, round(worst, 4), round(fl, 3), 'p1=%.1f d1=%.3f d2=%.3f p2=%.1f' % (np.degrees(x[15]), x[14], x[16], np.degrees(x[17])))
            if ok:
                R2 = fk_arm(x[7:14])[:3, :3]
                out[b] = dict(max_dq=worst, pick_pitch_deg=float(np.degrees(x[15])), PICK_D=float(x[14]),
                    place_pitch_deg=float(np.degrees(x[17])), place_roll_deg=roll_deg(R2), place_tool_x=R2[:, 0].round(4).tolist(),
                    PLACE_D=float(x[16]), q_pick=x[:7].tolist(), q_hover=x[26:33].tolist(),
                    q_place_059=x[7:14].tolist(), q_place_069=x[19:26].tolist(), R_place=R2.tolist(), floor_min=fl,
                    seed_rule='q_seed(zc) = q_place_059 + (zc-0.59)/0.10*(q_place_069 - q_place_059); q = solve([PLACE_D,0,zc], R_place, [q_seed])')
                break
    json.dump(out, open('opt3_result.json', 'w'), indent=1)
