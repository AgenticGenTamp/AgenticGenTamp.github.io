from env_client import make_env
import numpy as np
env = make_env()
for seed in [0,1]:
    obs, info = env.reset(seed=seed)
    print("info", info)
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        feats = env.observation_space.type_features[o.type] if hasattr(env.observation_space,'type_features') else None
        print(name, o.type, obs.vec([o]).round(3))
print(env.max_steps)
env.close()
