from env_client import make_env
import numpy as np
env = make_env()
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print("seed",seed, "info", info)
    for name in sorted(obs.get_object_names(), key=str):
        o = obs.get_object_from_name(name) if isinstance(name,str) else name
        t = o.type if hasattr(o,'type') else None
        feats = env.observation_space.type_features[t] if t is not None else []
        print(" ", o, [ (f, round(float(obs.get(o,f)),4)) for f in feats])
env.close()
