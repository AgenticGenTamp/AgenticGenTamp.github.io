import sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
from run import run
def f(s):
    try:
        te, t, tot, obs = run(s)
        return s, te, t, tot
    except Exception as e:
        return s, "ERR " + repr(e)[:100], 0, 0
if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2])
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(f, range(a, b)))
    for r in res:
        if r[1] is not True: print("FAIL", r)
    print("success", sum(r[1] is True for r in res), "/", len(res), "mean steps", np.mean([r[2] for r in res]))
