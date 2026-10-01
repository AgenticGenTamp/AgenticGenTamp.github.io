import os
import sys, numpy as np, json
from multiprocessing import Pool
import approach
from env_client import make_env
def run(args):
    seed, count, params = args
    if 'CFG' in params:
        import cfgs
        for k, v in getattr(cfgs, params['CFG']).items(): setattr(approach, k, v)
    for k, v in params.items(): setattr(approach, k, v)
    env = make_env()
    obs, info = env.reset(seed=seed, options={'object_count': count})
    ap = approach.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    n = 0; term = False
    while n < 1000:
        obs, r, term, trunc, info = env.step(ap.get_action(obs)); n += 1
        if term: break
    env.close()
    return seed, count, term, n
if __name__ == '__main__':
    params = json.loads(sys.argv[1]); counts = [int(c) for c in sys.argv[2].split(',')]
    seeds = range(int(sys.argv[3]), int(sys.argv[4]))
    with Pool(int(sys.argv[5]) if len(sys.argv) > 5 else 8) as p:
        res = p.map(run, [(s, c, params) for c in counts for s in seeds])
    print(params, 'solved %d/%d' % (sum(r[2] for r in res), len(res)), 'mean steps %.0f' % np.mean([r[3] for r in res]), 'fails', [(r[0], r[1]) for r in res if not r[2]])
    if os.environ.get('V'):
        for c in counts: print(c, [r[3] for r in res if r[1] == c])
