import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
cnt={}; cur=None; start=0
for i in range(1000):
    a=ap.get_action(obs)
    key=(ap.op[0] if ap.op else None, ap.phase)
    cnt[key]=cnt.get(key,0)+1
    obs,r,t,tr,info=env.step(a)
    if t: break
print('total',i+1)
agg={}
for (op,ph),v in cnt.items(): agg.setdefault(op,{})[ph]=v
for op in sorted(agg,key=lambda x:(x is None,x)): print(op, agg[op], sum(agg[op].values()))
env.close()
