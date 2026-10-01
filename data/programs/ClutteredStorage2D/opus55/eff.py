import sys, math, numpy as np
from env_client import make_env
from approach import GeneratedApproach
rows=[]
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    env=make_env(); obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    prev=None; start=None
    for t in range(1000):
        a=ap.get_action(obs)
        key=(ap.task[0] if ap.task else None, ap.phase)
        if key!=prev:
            if prev in (('fetch','nav'),('fetch','carry'),('push','nav')) and start:
                t0,x0,y0,th0=start
                lb=max(abs(ap.rx-x0),abs(ap.ry-y0))/0.05
                lr=abs(math.remainder(ap.rth-th0,2*math.pi))/(math.pi/16)
                rows.append((seed,prev[0]+'-'+prev[1],t-t0,round(max(lb,lr),1),round(lb,1),round(lr,1)))
            start=(t,ap.rx,ap.ry,ap.rth); prev=key
        obs,r,term,trunc,info=env.step(a)
        if term: break
    env.close()
import collections
agg=collections.defaultdict(lambda:[0,0])
for r in rows:
    agg[r[1]][0]+=r[2]; agg[r[1]][1]+=r[3]
    if r[2]-r[3]>6: print(r)
print(dict(agg))
