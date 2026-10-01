import sys, numpy as np
from env_client import make_env
G=float(sys.argv[1])
env=make_env(); obs,_=env.reset(seed=0); obs=np.asarray(obs,float)
rng=np.random.default_rng(11)
seq=rng.uniform(-0.1,0.1,(250,10))
for k in range(250):
    a=np.zeros(11); a[0:10]=seq[k]; a[10]=G
    o,r,te,tr,i=env.step(a); obs=np.asarray(o,float)
print(f"G={G} objA={np.round(obs[0:3],5)} objB={np.round(obs[16:19],5)} objC={np.round(obs[32:35],5)} q={np.round(obs[96:103],5)}")
env.close()
