import numpy as np
from env_client import make_env
env=make_env(); obs,info=env.reset(seed=0)
for n in sorted(obs.get_object_names()):
    o=obs.get_object_from_name(n)
    try: feats=obs.get_features(o) if hasattr(obs,'get_features') else None
    except Exception: feats=None
    print(n, o.type if hasattr(o,'type') else '', )
print(type(obs), [m for m in dir(obs) if not m.startswith('_')])
for n in sorted(obs.get_object_names()):
    o=obs.get_object_from_name(n)
    print(n,[ (f,round(float(obs.get(o,f)),4)) for f in obs.type_features[o.type]])
