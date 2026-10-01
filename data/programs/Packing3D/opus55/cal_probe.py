from env_client import make_env
import numpy as np
env = make_env()
for seed in range(6):
    obs, info = env.reset(seed=seed)
    tf = env.observation_space.type_features
    for o in obs.get_object_names():
        o = obs.get_object_from_name(o) if isinstance(o,str) else o
        print(seed, o, {k: round(float(obs.get(o,k)),3) for k in tf[o.type]})
