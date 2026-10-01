import numpy as np
from env_client import make_env
env=make_env()
obs,_=env.reset(seed=0); j0=np.asarray(obs)[96:103].copy()
rng=np.random.default_rng(3)
cum=np.zeros(7)
for t in range(60):
    a=np.zeros(11,dtype=np.float32); a[3:10]=rng.uniform(-0.1,0.1,7); cum+=a[3:10]
    obs,_,_,_,_=env.step(a)
z=np.zeros(11,dtype=np.float32)
for t in range(30): obs,_,_,_,_=env.step(z)
j1=np.asarray(obs)[96:103]
print("realized/expected(0.25*cum):", np.round((j1-j0)/(0.25*cum),4))
env.close()
