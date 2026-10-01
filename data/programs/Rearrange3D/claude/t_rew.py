import numpy as np
from env_client import make_env
env = make_env()
obs,info=env.reset(seed=0)
print("reset info", info)
rng=np.random.default_rng(0)
rs=set(); terms=[]
for t in range(200):
    a=np.concatenate([rng.uniform(-0.1,0.1,10), rng.uniform(0,1,1)]).astype(np.float32)
    obs,r,te,tr,info=env.step(a); rs.add(round(float(r),6))
    if te or tr: terms.append((t,te,tr)); break
print("rewards seen:", sorted(rs))
print("term/trunc:", terms, "info", info)
# zero-action 200
obs,_=env.reset(seed=0); rs=set()
for t in range(200):
    obs,r,te,tr,info=env.step(np.zeros(11,dtype=np.float32)); rs.add(round(float(r),6))
print("zero-action rewards:", sorted(rs))
env.close()
