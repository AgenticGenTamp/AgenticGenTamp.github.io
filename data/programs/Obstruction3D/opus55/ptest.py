import sys, time
from multiprocessing import Pool
from test_approach import run


def f(s):
    t = time.time()
    try:
        ok, n, tc, ap = run(s)
        return (s, ok, n, round(tc, 1), round(time.time() - t), ap.phase,
                ap.task['name'] if ap.task else None, len(ap._obstructions()))
    except Exception as e:
        return (s, False, -1, 0, 0, 'EXC', repr(e)[:200], 0)


if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2])
    npar = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    res = []
    with Pool(npar) as p:
        for r in p.imap_unordered(f, range(a, b)):
            res.append(r)
            print(('ok  ' if r[1] else 'FAIL'), r, flush=True)
    ok = [r for r in res if r[1]]
    print(f'success {len(ok)}/{len(res)}  mean steps(success) '
          f'{sum(r[2] for r in ok) / max(1, len(ok)):.1f}  max compute '
          f'{max(r[3] for r in res):.1f}s', flush=True)
