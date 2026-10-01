import sys
from multiprocessing import Pool
from test_approach import run
def f(args):
    sd,cnt=args
    try: return (sd,cnt)+tuple(run(sd,count=cnt))
    except Exception as ex: return (sd,cnt,'ERR',repr(ex)[:200])
if __name__=='__main__':
    a,b=int(sys.argv[1]),int(sys.argv[2]); cnt=int(sys.argv[3]) if len(sys.argv)>3 else None
    with Pool(12) as p:
        res=p.map(f,[(s,cnt) for s in range(a,b)])
    ok=[r for r in res if r[2] is True]
    print('solved',len(ok),'/',len(res),'mean steps',sum(r[3] for r in ok)/max(1,len(ok)),'max time',max(r[4] for r in res if len(r)>4))
    for r in res:
        if r[2] is not True: print(r)
