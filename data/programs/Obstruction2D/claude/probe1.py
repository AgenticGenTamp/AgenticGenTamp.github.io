from env_client import make_env
import numpy as np
env = make_env()
print("max_steps", env.max_steps)
print("action", env.action_space.low, env.action_space.high)
for seed in [0,1,42]:
    obs, info = env.reset(seed=seed)
    print("=== seed", seed, "info", info)
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        feats = {f: round(float(obs.get(o,f)),4) for f in obs.type_features[o.type]}
        print(name, o.type.name, feats)
env.close()
