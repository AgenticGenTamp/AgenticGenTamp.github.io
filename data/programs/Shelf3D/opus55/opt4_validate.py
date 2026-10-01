import numpy as np, pickle, glob, json, warnings
warnings.filterwarnings('ignore')
from approach import fk_arm, solve
from opt2_search import cab_margins
from opt3_validate import minz_path, roll_deg
from opt3_search import ZA, ZB
from opt4_search import branch_bounds, pick_link_m, place_link_m
def validate(x, v):
    lo, hi = branch_bounds(v)
    q1, qa, d1, p1, d2, p2, t, qb, qh = x[:7], x[7:14], x[14], x[15], x[16], x[17], x[18], x[19:26], x[26:33]
    R2 = fk_arm(qa)[:3, :3]
    inb = lambda q: np.all(q >= lo-1e-6) and np.all(q <= hi+1e-6)
    ok = inb(q1) and inb(qh) and pick_link_m(q1).min() > -1e-3 and pick_link_m(qh).min() > -1e-3
    fl = min(minz_path(q1, qh), 1.0); worst = 0
    for zc in np.linspace(ZA, ZB, 11):
        sd = qa + (zc - ZA)/(ZB - ZA)*(qb - qa)
        q, e = solve(np.array([d2, 0, zc]), R2, [sd])
        m = min(cab_margins(q, d2, zc).min(), place_link_m(q, d2, zc).min())
        worst = max(worst, np.abs(q - q1).max())
        fl = min(fl, minz_path(q1, q), minz_path(qh, q))
        ok &= bool(e < 2e-3 and m > -1e-3 and inb(q))
    ok &= fl > 0.0
    return bool(ok), worst, fl
if __name__ == '__main__':
    out = {}
    for v in ['R', 'W']:
        res = []
        for f in glob.glob('opt4_res_%s_*.pkl' % v): res += pickle.load(open(f, 'rb'))
        res.sort(key=lambda a: a[0]); print('==', v, len(res))
        last = -1; best_w = 9; cnt = 0; n_seen = 0
        for t, x in res:
            if abs(t - last) < 1e-4: continue
            last = t; n_seen += 1; ok, worst, fl = validate(x, v)
            print(round(t, 4), ok, round(worst, 4), round(fl, 3), 'p1=%.1f d1=%.3f d2=%.3f p2=%.1f' % (np.degrees(x[15]), x[14], x[16], np.degrees(x[17])))
            if ok and worst < best_w:
                best_w = worst; cnt = 0
                R2 = fk_arm(x[7:14])[:3, :3]
                out[v] = dict(max_dq=worst, pick_pitch_deg=float(np.degrees(x[15])), PICK_D=float(x[14]),
                    place_pitch_deg=float(np.degrees(x[17])), place_roll_deg=roll_deg(R2), place_tool_x=R2[:, 0].round(4).tolist(),
                    PLACE_D=float(x[16]), q_pick=x[:7].tolist(), q_hover=x[26:33].tolist(),
                    q_place_059=x[7:14].tolist(), q_place_069=x[19:26].tolist(), R_place=R2.tolist(), floor_min=fl,
                    joint_box=[branch_bounds(v)[0].tolist(), branch_bounds(v)[1].tolist()],
                    seed_rule='q_seed(zc) = q_place_059 + (zc-0.59)/0.10*(q_place_069 - q_place_059); q = solve([PLACE_D,0,zc], R_place, [q_seed])')
                pass
            cnt = cnt + 1 if ok else cnt
            if t > best_w - 1e-9 or n_seen > 40: break
    json.dump(out, open('opt4_result.json', 'w'), indent=1)
