import numpy as np, sys, json
from env_client import make_env
mode=sys.argv[1]; seed=int(sys.argv[2]); oc=int(sys.argv[3]); n=int(sys.argv[4])
env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':oc})
print(mode,seed,oc,'info0 keys',sorted(info.keys()) if hasattr(info,'keys') else type(info),flush=True)
rs={}; term=None
rng=np.random.default_rng(seed)
for t in range(n):
    if mode=='rand': a=env.action_space.sample()
    elif mode=='zero': a=np.zeros(11,dtype=np.float32)
    elif mode=='max': a=np.full(11,0.1,dtype=np.float32); a[10]=1.0
    obs,r,te,tr,i=env.step(a)
    r=float(r); rs[round(r,6)]=rs.get(round(r,6),0)+1
    if te or tr: term=(t,te,tr); break
print(mode,seed,oc,'rewards',rs,'term',term,'infoN',sorted(i.keys()) if hasattr(i,'keys') else i,flush=True)
env.close()
