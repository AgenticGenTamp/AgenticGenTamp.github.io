import sys, os
from multiprocessing import Pool
from test_approach import run
import functools, approach
for kv in filter(None, os.environ.get('OVR', '').split(',')):
    k, v = kv.split('='); setattr(approach, k, eval(v))
if __name__=='__main__':
    a,b=int(sys.argv[1]),int(sys.argv[2])
    with Pool(10) as p:
        cnt=int(sys.argv[3]) if len(sys.argv)>3 else None
        res=p.map(functools.partial(run,count=cnt), range(a,b))
    for r in res: print(r)
    print('solved', sum(r['term'] for r in res),'/',len(res), 'mean steps', sum(r['steps'] for r in res)/len(res))
