import sys, numpy as np
from multiprocessing import Pool
from test_approach import run
seeds=list(range(int(sys.argv[1]),int(sys.argv[2])))
with Pool(min(12,len(seeds))) as p:
    rs=p.map(lambda s: None, []) or p.map(run.__call__, seeds) if False else p.map(run, seeds)
ok=sum(r['term'] for r in rs)
print('solved',ok,'/',len(rs),'mean steps',np.mean([r['steps'] for r in rs]))
for r in rs:
    if not r['term']: print('FAIL',r)
