import sys
from multiprocessing import Pool
from test_approach import run
def job(a):
    s,c,k=a
    try: return (s,c)+run(s,c)
    except Exception as e: return (s,c,False,0,0,str(e))
if __name__=="__main__":
    seeds=[int(x) for x in sys.argv[1].split(',')]; c=int(sys.argv[2]); reps=int(sys.argv[3])
    jobs=[(s,c,k) for s in seeds for k in range(reps)]
    with Pool(int(sys.argv[4])) as p: res=p.map(job,jobs,chunksize=1)
    import numpy as np
    for s in seeds:
        rr=[r for r in res if r[0]==s]
        print(s, sum(r[2] for r in rr),"/",len(rr), "steps", [r[3] for r in rr], "t", [round(r[4]) for r in rr])
    print("total", sum(r[2] for r in res), len(res))
