from env_client import make_env
import numpy as np
env = make_env()
FEAT = {t['name']: t['features'] for t in __import__('json').load(open('env_spaces.json'))['observation_space']['types']}
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print("=== seed", seed, info)
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        feats = FEAT[o.type.name]
        print(" ", name, {f: round(float(obs.get(o,f)),4) for f in feats})
