import sys, numpy as np
from run_test import env
from approach import GeneratedApproach, wrap
seed=int(sys.argv[1]); n=int(sys.argv[2]); T0,T1=int(sys.argv[3]),int(sys.argv[4])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
for t in range(T1):
    a=ap.get_action(s); q=ap.queue[0]
    s,r,term,tr,_=env.step(a)
    if t>=T0:
        b,arm,g,cubes,_=ap._read(s)
        print(t,q.name,q.steps,'berr',(q.base-b).round(3) if q.base is not None else None,'aerr',np.round(np.max(np.abs(wrap(q.arm-arm))),3),'act',a[:3].round(3))
    if term: break
print("base", b.round(3)); print("cubes", [c.round(3).tolist() for c in cubes]); print("wiper", ap.wiper.round(3)); print("tgt", q.base)
