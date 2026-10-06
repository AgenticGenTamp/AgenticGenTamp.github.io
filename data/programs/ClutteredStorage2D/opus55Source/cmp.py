import sys
from multiprocessing import Pool
import approach
from test_approach import run
def f(a):
    n, s, flag = a
    approach.GeneratedApproach.ENABLE_STAGING = flag; approach.GeneratedApproach.STAGE_MIN_GAIN = float(sys.argv[3])
    return run(s, n)[1]
if __name__ == "__main__":
    n = int(sys.argv[1]); seeds = range(int(sys.argv[2]))
    with Pool(16) as p:
        on = p.map(f, [(n, s, True) for s in seeds])
        off = p.map(f, [(n, s, False) for s in seeds])
    for s, a, b in zip(seeds, on, off):
        if a != b: print(s, "on", a, "off", b)
    import numpy as np
    print("mean on", np.mean(on), "off", np.mean(off))
