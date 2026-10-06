import sys
from multiprocessing import Pool
from test_approach import run
def f(a):
    n, s = a
    try:
        return (n, s) + run(s, n)
    except Exception as e:
        return (n, s, False, -1, repr(e)[:100])
if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1].split(",")]
    seeds = range(int(sys.argv[2]), int(sys.argv[3]))
    jobs = [(n, s) for n in ns for s in seeds]
    with Pool(16) as p:
        res = p.map(f, jobs, chunksize=1)
    fails = [r for r in res if not r[2]]
    for r in fails: print("FAIL", r)
    for r in res:
        if isinstance(r[4], float) and r[4] > 15: print("SLOW", r)
    import statistics
    for n in ns:
        rr = [r for r in res if r[0]==n]
        print(n, sum(r[2] for r in rr), "/", len(rr), "mean steps", statistics.mean(r[3] for r in rr if r[2]) if any(r[2] for r in rr) else None, "max time", max(r[4] for r in rr if isinstance(r[4], float)))
