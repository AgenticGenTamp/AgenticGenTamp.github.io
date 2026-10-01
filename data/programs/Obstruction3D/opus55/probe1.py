from env_client import make_env
import numpy as np
env = make_env()
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print('seed',seed,info)
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        feats = obs.type_features[o.type] if isinstance(obs.type_features, dict) else None
        vals = {f: round(float(obs.get(o,f)),4) for f in feats}
        print(' ',name, vals)
