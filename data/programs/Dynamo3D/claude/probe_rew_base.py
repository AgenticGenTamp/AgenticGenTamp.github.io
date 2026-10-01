from env_client import make_env
import numpy as np
for seed in (1,11):
    env=make_env(); obs,info=env.reset(seed=seed)
    print("seed",seed,"info",info)
    for nm in sorted(obs.get_object_names()):
        o=obs.get_object_from_name(nm)
        print(" obj",nm,o.type.name)
        d={f:round(float(obs.get(o,f)),3) for f in obs.type_features[o.type]}
        print("  ",d)
    a=np.zeros(11,dtype=np.float32)
    obs,r,term,trunc,info=env.step(a)
    print(" idle step r",r,term,trunc,info)
    env.close()
