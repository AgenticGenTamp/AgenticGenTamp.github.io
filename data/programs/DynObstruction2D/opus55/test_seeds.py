import sys
from test_approach import run
from concurrent.futures import ThreadPoolExecutor
import numpy as np
seeds=[int(s) for s in sys.argv[1:]]
with ThreadPoolExecutor(8) as ex: res=list(ex.map(run, seeds))
for s,term,steps,oc,dt in res: print(s, 'OK' if term else 'FAIL', steps, '%.1fs'%dt)
print('success %d/%d'%(sum(r[1] for r in res),len(res)))
