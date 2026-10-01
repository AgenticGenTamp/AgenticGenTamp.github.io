from env_client import make_env
import numpy as np, collections
env = make_env()
cnt = collections.Counter(); xs=[];ys=[]
for seed in range(40):
    obs, info = env.reset(seed=seed)
    cnt[info.get('object_count')]+=1
    for o in obs.data:
        xs.append(obs.get(o,'x')); ys.append(obs.get(o,'y'))
print(cnt); print(min(xs),max(xs),min(ys),max(ys))
env.close()
