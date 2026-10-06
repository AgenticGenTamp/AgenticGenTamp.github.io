import sys
from multiprocessing import Pool
from test_approach import run
def job(a):
    s,c=a
    try: return (s,c)+run(s,c)
    except Exception as e: return (s,c,False,0,0,str(e))
if __name__=="__main__":
    counts=[int(c) for c in sys.argv[1].split(',')]
    seeds=range(int(sys.argv[2]), int(sys.argv[3]))
    jobs=[(s,c) for c in counts for s in seeds]
    with Pool(int(sys.argv[4]) if len(sys.argv)>4 else 8) as p:
        res=p.map(job,jobs,chunksize=1)
    ok=0
    for r in res:
        ok+=r[2]
        if not r[2] or r[4]>20: print("FAIL/SLOW",r)
    import numpy as np
    print("success",ok,"/",len(res), "mean steps(succ)", np.mean([r[3] for r in res if r[2]]), "max time", max(r[4] for r in res))
    for c in counts:
        rr=[r for r in res if r[1]==c]
        print("count",c,"succ",sum(r[2] for r in rr),"/",len(rr),"mean steps",round(np.mean([r[3] for r in rr]),1),"max steps",max(r[3] for r in rr),"mean t",round(np.mean([r[4] for r in rr]),2))
