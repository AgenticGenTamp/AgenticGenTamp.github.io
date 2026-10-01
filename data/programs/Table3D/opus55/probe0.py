from env_client import make_env
import numpy as np
env = make_env()
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print("seed",seed,"info",info)
    for t in env.observation_space.types:
        for o in obs.get_objects(t):
            print(o.name, t.name if hasattr(t,'name') else t, {f: round(float(obs.get(o,f)),3) for f in env.observation_space.type_features[t]})
print(env.max_steps)
env.close()
