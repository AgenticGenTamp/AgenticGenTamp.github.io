from env_client import make_env
import numpy as np
env = make_env()
feats = {t['name']: t['features'] for t in __import__('json').load(open('env_spaces.json'))['observation_space']['types']}
def dump(obs):
    for n in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(n)
        tn = o.type.name if hasattr(o.type,'name') else str(o.type)
        print(n, {f: round(float(obs.get(o,f)),3) for f in feats[tn]})
for seed in range(4):
    obs, info = env.reset(seed=seed)
    print("seed", seed, info)
    dump(obs)
env.close()
