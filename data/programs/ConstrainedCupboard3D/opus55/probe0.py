from env_client import make_env
import numpy as np
env = make_env()
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print("seed",seed,"info",info)
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        feats = env.observation_space.type_features[o.type] if hasattr(o,'type') else None
        vals = {f: round(obs.get(o,f),3) for f in feats}
        print(name, vals)
env.close()
