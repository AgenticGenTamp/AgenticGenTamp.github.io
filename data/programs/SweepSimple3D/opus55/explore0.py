from env_client import make_env
import numpy as np
env = make_env()
for seed in [0,1]:
    obs, info = env.reset(seed=seed)
    print("seed",seed, "info", info)
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        t = o.type.name if hasattr(o,'type') else '?'
        feats = env.observation_space.get_type(t) if False else None
        try:
            fs = env.observation_space.type_features[o.type] 
        except Exception:
            fs = None
        vals = {}
        for tt in env.observation_space.types:
            if tt.name==t:
                fs = env.observation_space.type_features[tt]
        print(name, t, {f: round(float(obs.get(o,f)),3) for f in fs})
env.close()
