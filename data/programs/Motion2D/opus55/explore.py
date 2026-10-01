from env_client import make_env
import numpy as np
env = make_env()
for seed in range(4):
    obs, info = env.reset(seed=seed)
    print("seed", seed, info)
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        t = o.type if hasattr(o,'type') else None
        print(" ", name, t)
        tn = getattr(t,'name',t)
        feats = env.observation_space.type_features[t] if t in env.observation_space.type_features else None
        if feats is None:
            for k,v in env.observation_space.type_features.items():
                if getattr(k,'name',k)==tn: feats=v
        print("    ", {f: round(float(obs.get(o,f)),3) for f in feats})
env.close()
