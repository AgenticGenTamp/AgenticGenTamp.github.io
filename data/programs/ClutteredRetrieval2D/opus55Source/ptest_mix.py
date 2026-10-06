import sys
from multiprocessing import Pool
from ptest import job
if __name__=="__main__":
    a,b=int(sys.argv[1]),int(sys.argv[2]); mx=int(sys.argv[3])
    jobs=[(s,s%(mx+1)) for s in range(a,b)]
    with Pool(int(sys.argv[4])) as p: res=p.map(job,jobs,chunksize=1)
    import numpy as np
    for r in res:
        if not r[2] or r[4]>30: print("FAIL/SLOW",r)
    print("success",sum(r[2] for r in res),"/",len(res),"mean steps",np.mean([r[3] for r in res if r[2]]),"mean time",np.mean([r[4] for r in res]))
