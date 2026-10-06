import sys, time
from multiprocessing import Pool
from test_approach import run
import os, approach
for kv in os.environ.get("APX", "").split(","):
    if kv:
        k, v = kv.split("=")
        setattr(approach.GeneratedApproach, k, eval(v))
def f(args):
    sd, c = args
    try:
        ok, n, ph = run(sd, c)
    except Exception as e:
        ok, n, ph = False, -1, "EXC " + repr(e)[:80]
    return c, sd, ok, n, ph
if __name__ == "__main__":
    counts = [int(c) for c in sys.argv[1].split(",")]
    s0, s1 = int(sys.argv[2]), int(sys.argv[3])
    jobs = [(sd, c) for c in counts for sd in range(s0, s1)]
    with Pool(16) as p:
        res = p.map(f, jobs)
    fails = [(c, sd, n, ph) for c, sd, ok, n, ph in res if not ok]
    for c in counts:
        rc = [r for r in res if r[0] == c]
        print(f"count {c}: {sum(r[2] for r in rc)}/{len(rc)}  mean steps(succ) {sum(r[3] for r in rc if r[2])/max(1,sum(r[2] for r in rc)):.0f}  mean all {sum(r[3] if r[2] else 300+150*c for r in rc)/len(rc):.1f}")
    print("fails:", fails)
