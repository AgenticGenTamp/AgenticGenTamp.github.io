import sys, numpy as np, approach
from test_approach import run
approach.ORDER=float(sys.argv[2])
r=run(int(sys.argv[1]),False)
print({k:(round(float(v),4) if not isinstance(v,(bool,np.bool_)) else v) for k,v in r.items()})
