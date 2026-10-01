import sys, numpy as np
from multiprocessing import Pool
from test_approach import run
seeds=[int(x) for x in sys.argv[1:]]
with Pool(min(12,len(seeds))) as p:
    rs=p.map(run, seeds)
for r in rs: print({k:(round(float(v),3) if not isinstance(v,(bool,np.bool_)) else v) for k,v in r.items()})
