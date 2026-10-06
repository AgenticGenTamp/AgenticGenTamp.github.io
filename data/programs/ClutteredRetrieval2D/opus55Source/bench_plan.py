import sys, time, pickle, glob, numpy as np
from multiprocessing import Pool
import approach
from approach import *
VAR=sys.argv[1]
def rrt_restart(chk, start, goals, rng, budget):
    t0=time.time()
    slices=[0.3,0.3,0.4] if VAR=='restart' else [1.0]
    for f in slices:
        rem=budget-(time.time()-t0)
        if rem<=0.02: break
        p=rrt_connect(chk,start,goals,rng,time_limit=min(rem,budget*f))
        if p is not None: return p
    return None
def job(fn):
    Q=pickle.load(open(fn,'rb')); out=[]
    rng=np.random.default_rng(1)
    for chk,start,goals,tb,ok,dt in Q:
        goals=[g for g in goals if chk.free(g)]
        if not goals: continue
        if not chk.free(start): continue
        goals.sort(key=lambda g: nsteps(start,g)); goals=goals[:60]
        for rep in range(3 if VAR!="long" else 1):
            t=time.time(); p=rrt_restart(chk,start,goals,rng,min(tb,3.0) if VAR!="long" else 10.0); out.append((p is not None, time.time()-t))
    return out
with Pool(12) as P:
    R=sum(P.map(job, sorted(glob.glob('logs/q_*.pkl'))),[])
print(VAR, 'n',len(R),'succ',sum(r[0] for r in R),'time',round(sum(r[1] for r in R),1), 'succ-time', round(sum(r[1] for r in R if r[0]),1))
