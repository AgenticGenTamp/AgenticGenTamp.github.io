import numpy as np
from env_client import make_env
env=make_env()
rs=set()
for seed in [0,3]:
    obs,info=env.reset(seed=seed)
    rng=np.random.default_rng(seed)
    for t in range(400):
        a=np.concatenate([rng.uniform(-0.1,0.1,10),[float(rng.random()<0.5)]]).astype(np.float32)
        obs,r,te,tr,info=env.step(a); rs.add(round(float(r),6))
        if te or tr: print("ENDED",t,te,tr); break
print("rewards:", sorted(rs), "info keys:", list(info.keys()))
env.close()
