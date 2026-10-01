import numpy as np
from env_client import make_env
np.set_printoptions(precision=4,suppress=True,linewidth=200)
env=make_env()
def A(b=(0,0,0),arm=(0,)*7,g=0.0):
    return np.array(list(b)+list(arm)+[g],dtype=np.float32)
R=lambda o:o[93:115]
rng=np.random.default_rng(0)
obs,_=env.reset(seed=0); rs=[]; infos=set()
for k in range(250):
    a=np.concatenate([rng.uniform(-0.1,0.1,10),rng.uniform(0,1,1)]).astype(np.float32)
    obs,r,te,tr,info=env.step(a); rs.append(r); infos.add(str(sorted(info.keys())))
    if te or tr: print('END at',k,te,tr,info); break
print('random: unique rewards',np.unique(rs),'n',len(rs),'infos',infos)
# j2 + with base retreated
obs,_=env.reset(seed=0)
for k in range(8): obs,*_=env.step(A(b=(-0.1,0,0)))
print('base',R(obs)[:3])
for k in range(100): obs,*_=env.step(A(arm=(0,0.1,0,0,0,0,0)))
print('j2 + far from table:',R(obs)[4])
