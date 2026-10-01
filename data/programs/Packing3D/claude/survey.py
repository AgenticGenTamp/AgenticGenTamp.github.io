import time,numpy as np
from env_client import make_env
import pl
env=make_env(); d=pl.Drv(env)
t0=time.time()
for sd in range(10):
    for oc in (1,):
        o=d.reset(seed=sd,oc=oc)
        ns=pl.parts(o)
        info=[]
        for n in ns:
            f=pl.feat(o,n)
            tt=f.get('triangle_type',None)
            info.append((n,'tri%d'%tt if tt is not None else 'cub',np.round(pl.ppos(o,n),3),round(pl.yaw(o,n),3),sorted(f.keys())if sd==0 else ''))
        print(sd,oc,info,flush=True)
print('rack',pl.feat(d.obs,'rack'))
print('robot',pl.feat(d.obs,'robot'))
print('objs',d.obs.get_object_names())
print('t',time.time()-t0)
env.close()
