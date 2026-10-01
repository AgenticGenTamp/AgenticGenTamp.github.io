import sys, numpy as np
from multiprocessing import Pool
from test_approach import run
def job(args):
    seed, count = args
    try:
        term, steps, dt, fin, cz = run(seed, count)
        return seed, count, term, steps, dt, fin
    except Exception as e:
        return seed, count, False, -1, 0, str(e)[:200]
if __name__ == '__main__':
    counts = [int(c) for c in sys.argv[1].split(',')]
    seeds = range(int(sys.argv[2]), int(sys.argv[3]))
    jobs = [(s, c) for c in counts for s in seeds]
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 8) as p:
        res = p.map(job, jobs)
    ok = 0
    for seed, count, term, steps, dt, fin in res:
        ok += term
        if not term or '-v' in sys.argv:
            print('FAIL' if not term else 'ok', seed, count, steps, '%.1f' % dt, fin)
    for c in counts:
        r = [x for x in res if x[1] == c]
        print('count', c, 'solved %d/%d' % (sum(x[2] for x in r), len(r)), 'mean steps %.0f' % np.mean([x[3] for x in r if x[2]] or [0]), 'max time %.1f' % max(x[4] for x in r))
