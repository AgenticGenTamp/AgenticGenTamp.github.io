import sys
import approach
approach.VARIANTS = approach.VARIANTS + [("float","rand","float"),("park","rand","float"),("cost","rand","float"),("float","rand","float"),("park","rand","float"),("park","rand","cost")]
import varstat
from multiprocessing import Pool
import numpy as np
if __name__=="__main__":
    with Pool(40) as p:
        res=p.map(varstat.run,[(c,s) for c in [2,3,4] for s in range(0,40)])
    L=np.array([[o[0] for o in r[2]] for r in res])
    print("det8", L[:,:8].min(1).mean(), "all", L.min(1).mean(), "det8+3", L[:,:11].min(1).mean())
