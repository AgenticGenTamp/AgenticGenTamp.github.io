from env_client import make_env
import numpy as np
env = make_env()
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print("seed",seed, "info", info)
    for t in ["robot","surface","block"]:
        for o in obs.get_objects(env.observation_space.get_type(t)) if hasattr(env.observation_space,'get_type') else []:
            feats = env.observation_space.type_features[env.observation_space.get_type(t)]
            print(o.name, {f: round(float(obs.get(o,f)),4) for f in feats})
env.close()
