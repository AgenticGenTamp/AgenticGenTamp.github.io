import sys
import numpy as np
from env_client import make_env
import toss as T
seed = int(sys.argv[1]); nobj = int(sys.argv[2]) if len(sys.argv) > 2 else 2
env = make_env()
obs, _ = env.reset(seed=seed, options={'object_count': nobj})
def q(o, nm):
    C = o.get_object_from_name(nm); return np.array([o.get(C, f) for f in ['qw','qx','qy','qz']])
g = T.toss_all(obs); a = next(g)
for t in range(1, 400):
    obs, r, te, tr, _ = env.step(a)
    b = T.objpos(obs, 'bin_0')
    rels = [(T.objpos(obs, n) - b).round(3).tolist() for n in T.cube_names(obs)]
    if any(abs(x[0]) < 0.3 and x[2] < 0.3 for x in rels) and (t % 5 == 0 or te or any(abs(x[0]) < 0.3 and x[2] < 0.3 and abs(x[2]) > 0.0 for x in rels) and t < 0):
        print(t, te, 'binz %.3f' % b[2], 'bq', q(obs, 'bin_0').round(3), 'rels', rels)
    if te: break
    a = g.send(obs)
env.close()
