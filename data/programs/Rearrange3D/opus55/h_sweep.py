import sys, numpy as np
from multiprocessing import Pool
import approach
from test_approach import run
def f(args):
    h,d,seed=args
    approach.PLACE_H=h; approach.PLACE_DIST[approach.CAN]=d
    r=run(seed,verbose=False)
    return h,d,seed,r['term'],r['steps'],round(float(r['d_can']),3),round(float(r['zc']),3)
jobs=[(h,0.12,3) for h in [0.03,0.06,0.12]]+[(0.005,d,3) for d in [0.165,0.17,0.175]]
with Pool(len(jobs)) as p:
    for x in p.map(f,jobs): print(x)
