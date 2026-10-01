import sys, time, json
from concurrent.futures import ThreadPoolExecutor
from test_approach import run
seeds=list(range(int(sys.argv[1]),int(sys.argv[2])))
t0=time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    res=list(ex.map(lambda s: run(s), seeds))
bad=[r for r in res if not r['term']]
import numpy as np
print('solved %d/%d'%(len(res)-len(bad),len(res)),'mean steps %.1f'%np.mean([r['steps'] for r in res]),'max wall %.1f'%max(r['wall'] for r in res))
for r in sorted(res,key=lambda r:-r['steps'])[:12]: print(r)
print('elapsed',round(time.time()-t0,1))
