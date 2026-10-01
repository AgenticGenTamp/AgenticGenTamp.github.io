import sys
from multiprocessing import Pool
def work(seeds):
    from env_client import make_env
    from test_approach import run
    env=make_env(); out=[]
    for s in seeds:
        succ,n,ph=run(s,env); out.append((s,succ,n,' '.join(f'{p}@{t}' for p,t in ph)))
    env.close(); return out
if __name__=='__main__':
    a,b=int(sys.argv[1]),int(sys.argv[2]); P=int(sys.argv[3]) if len(sys.argv)>3 else 8
    import os
    seeds=[int(x) for x in os.environ['SEEDS'].split(',')] if os.environ.get('SEEDS') else list(range(a,b)); chunks=[seeds[i::P] for i in range(P)]
    with Pool(P) as p: res=sum(p.map(work,chunks),[])
    res.sort()
    fails=[r for r in res if not r[1]]
    for r in (res if os.environ.get('VERB') else fails): print(r[0], r[1], r[2], r[3][-300:])
    ok=[r[2] for r in res if r[1]]
    print('success',len(ok),'/',len(res),'mean steps',sum(ok)/max(1,len(ok)))
