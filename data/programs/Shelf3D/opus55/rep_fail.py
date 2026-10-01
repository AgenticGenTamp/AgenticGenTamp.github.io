import sys, os
from multiprocessing import Pool
def run(args):
    seed, count, rep = args
    from env_client import make_env
    import approach
    env = make_env()
    obs, info = env.reset(seed=seed, options={'object_count': count})
    ap = approach.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    seq = []; n = 0; term = False; cur = []
    while n < 1000:
        a = ap.get_action(obs)
        tag = ap.phase + ':' + str(getattr(ap, 'cur', ''))
        if not seq or seq[-1][0] != tag: seq.append([tag, 0])
        seq[-1][1] += 1
        obs, r, term, trunc, info = env.step(a); n += 1
        if term: break
    r, c, _ = ap._parse(obs)
    env.close()
    return seed, count, rep, term, n, seq, {k: v[:3].round(3).tolist() for k, v in c.items()}, r[:3].round(2).tolist()
if __name__ == '__main__':
    seeds = [int(x) for x in sys.argv[1].split(',')]; counts = [int(x) for x in sys.argv[2].split(',')]; reps = int(sys.argv[3])
    with Pool(5) as p:
        res = p.map(run, [(s, c, k) for s in seeds for c in counts for k in range(reps)])
    for s, c, k, term, n, seq, cubes, base in res:
        print(s, c, k, term, n)
        if not term:
            print('  ', ' '.join(f'{t[:3]}{t.split(":")[1][-1:]}:{m}' for t, m in seq)[:1200])
            print('  cubes', cubes, 'base', base)
