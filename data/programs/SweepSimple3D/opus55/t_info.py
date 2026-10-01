from env_client import make_env
import numpy as np
env=make_env()
for s in [1,6,9,11,14,0]:
    obs,info=env.reset(seed=s)
    out=[]
    for n in sorted(obs.get_object_names()):
        o=obs.get_object_from_name(n)
        f=obs.type_features[o.type]
        if 'x' in f: out.append((n,o.type.name,[round(obs.get(o,k),3) for k in ('x','y','z')]))
        else: out.append((n,o.type.name))
    print(s, info, out)
obs,r,te,tr,info=env.step(np.zeros(11,np.float32)); print(r,te,tr,info)
