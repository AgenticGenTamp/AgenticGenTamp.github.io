import numpy as np
from env_client import make_env
env=make_env()
rng=np.random.default_rng(0)
for seed in [0,3,7]:
    obs,info=env.reset(seed=seed)
    rews=[]; ends=None
    for t in range(400):
        a=np.concatenate([rng.uniform(-0.1,0.1,10), rng.uniform(0,1,1)]).astype(np.float32)
        obs,r,term,trunc,info=env.step(a); rews.append(r)
        if term or trunc: ends=(t,term,trunc); break
    o=np.asarray(obs,float)
    rews=np.array(rews)
    print(seed,"uniq r:",np.unique(np.round(rews,4))[:8],"end",ends)
    print("  blocks",np.round(o[0:3],3),np.round(o[54:57],3),np.round(o[70:73],3),"ss",np.round(o[38:41],3),np.round(o[41:45],3))
