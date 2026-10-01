import sys, numpy as np
from multiprocessing import Pool
import approach
from test_approach import run
from env_client import make_env
def job(args):
    seed,order=args
    approach.ORDER=order
    r=run(seed,False)
    return seed,order,r['steps']
def feats(seed):
    env=make_env(); o,_=env.reset(seed=seed); env.close()
    return o[0:3],o[16:18],o[32:34],o[93:96]
if __name__=='__main__':
    seeds=[s for s in range(0,40) if s not in (27,32,36,46)]
    with Pool(12) as p: rs=p.map(job,[(s,o) for s in seeds for o in (1.0,-1.0)])
    d={}
    for s,o,st in rs: d.setdefault(s,{})[o]=st
    for s in seeds:
        b,dr,ca,rb=feats(s)
        print(s,d[s][1.0],d[s][-1.0],'bowl',b[:2].round(2),'drink',dr.round(2),'can',ca.round(2),'rob',rb.round(2))
