import numpy as np, pickle, glob, sys, warnings
warnings.filterwarnings('ignore')
import opt2_search as S
from approach import fk_arm, solve, LIM
from scipy.optimize import minimize
case, horiz = sys.argv[1], sys.argv[2] == 'h'
p1fix = None if case.startswith('free') else 90.0
res = []
for f in glob.glob('opt2_old/opt2_res_%s_*.pkl' % case): res += pickle.load(open(f, 'rb'))
res.sort(key=lambda a: a[0])
# reuse run() internals by monkeypatching the initializer: simple re-implementation
out = []
last = -1
for t, x in res[:60]:
    if abs(t - last) < 1e-3: continue
    last = t
    q1, q2, d1, p1, d2, p2 = x[:7], x[7:14], x[14], x[15], x[16], x[17]
    R2 = fk_arm(q2)[:3, :3]
    qa, ea = solve(np.array([d2, 0, 0.61]), R2, [q2]); qb, eb = solve(np.array([d2, 0, 0.69]), R2, [q2])
    x0 = np.r_[q1, qa, d1, p1, d2, p2, max(np.abs(qa-q1).max(), np.abs(qb-q1).max()), qb]
    out.append(x0)
pickle.dump(out, open('opt2_x0_%s.pkl' % case, 'wb'))
