from env_client import make_env
import numpy as np
def d(obs,n):
    o=obs.get_object_from_name(n); return {f: float(obs.get(o,f)) for f in obs.type_features[o.type]}
env=make_env(); obs,info=env.reset(seed=0)
print(sorted(obs.get_object_names()))
for n in sorted(obs.get_object_names()):
    print(n, {k:round(v,4) for k,v in d(obs,n).items()})
print("max_steps",env.max_steps)
env.close()
