import sys, numpy as np
from multiprocessing import Pool
import approach
from test_approach import run
def f(args):
    d,seed=args
    approach.PLACE_DIST[approach.CAN]=d
    r=run(seed,verbose=False)
    return d,seed,r['term'],r['steps'],round(float(r['d_can']),3),round(float(r['d_drink']),3)
jobs=[(d,s) for d in [0.08,0.10,0.14,0.16,0.18,0.20,0.24] for s in [3]]
with Pool(len(jobs)) as p:
    for x in p.map(f,jobs): print(x)
