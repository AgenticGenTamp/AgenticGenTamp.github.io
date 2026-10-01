from env_client import make_env
import numpy as np
env = make_env()
for seed in range(6):
    obs, info = env.reset(seed=seed)
    print("=== seed", seed, "info", info)
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        feats = obs.type_features[o.type]
        print(" ", name, o.type.name, {f: round(float(obs.get(o,f)),4) for f in feats})
env.close()
