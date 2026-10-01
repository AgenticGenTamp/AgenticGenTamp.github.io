from env_client import make_env
import numpy as np
env = make_env()
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print("seed",seed, info)
    for n in obs.get_object_names():
        o = obs.get_object_from_name(n)
        t = o.type if hasattr(o,'type') else None
        tn = getattr(t,'name',t)
        feats = env.observation_space.type_features[o.type]
        print(n, tn, {f: round(float(obs.get(o,f)),3) for f in feats})
env.close()
