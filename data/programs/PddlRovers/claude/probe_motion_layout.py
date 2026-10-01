from env_client import make_env
import numpy as np
env = make_env()
print("max_steps", env.max_steps)
for seed in [0,1]:
    obs, info = env.reset(seed=seed)
    print("=== seed", seed, info)
    for n in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(n)
        feats = {f: round(float(obs.get(o,f)),3) for f in obs.type_features[o.type]}
        print(n, feats)
env.close()
