from env_client import make_env
import numpy as np
env = make_env()
for seed in range(3):
    obs, info = env.reset(seed=seed)
    print("seed", seed, info)
    for t in obs.get_object_names() if hasattr(obs,'get_object_names') else []:
        o = obs.get_object_from_name(t)
        print(t, o.type.name if hasattr(o.type,'name') else o.type, {f: round(float(obs.get(o,f)),3) for f in env.observation_space.get_type(o.type.name if hasattr(o.type,'name') else o.type).feature_names} if False else "")
    print(obs)
env.close()
