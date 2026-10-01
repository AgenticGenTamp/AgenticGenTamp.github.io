import sys, numpy as np
from multiprocessing import Pool
import approach
from test_approach import run as _run
import traceback
def run(seed):
    try:
        return _run(seed, False)
    except Exception as e:
        return dict(seed=seed, term=False, steps=1000, err=traceback.format_exc()[-300:])
def init(kv):
    for k,v in kv.items(): setattr(approach,k,v)
if __name__=='__main__':
    kv={}
    args=sys.argv[1:]
    seeds=[s for s in range(int(args[0]),int(args[1])) if s not in (27,32,36,46,88,138,49,53,60,101,125,146,51)]
    for a in args[2:]:
        k,v=a.split('='); kv[k]=float(v)
    with Pool(12,initializer=init,initargs=(kv,)) as p:
        rs=p.map(run,seeds)
    st=[r['steps'] for r in rs]; ok=sum(r['term'] for r in rs)
    for r in rs:
        if 'err' in r: print('ERR', r['seed'], r['err'])
        elif not r['term']: print('FAIL', {k: (round(float(v), 3) if not isinstance(v, (bool, np.bool_)) else v) for k, v in r.items()})
    print(kv,'solved',ok,'/',len(rs),'mean',np.mean(st),'fails',[r['seed'] for r in rs if not r['term']])
    good=[r for r in rs if r.get('term')]
    print('solved-mean',np.mean([r['steps'] for r in good]),'top',sorted([(int(r['steps']),r['seed']) for r in good],reverse=True)[:12])
