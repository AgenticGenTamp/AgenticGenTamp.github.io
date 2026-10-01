from env_client import make_env
import numpy as np
env = make_env()
def dump(obs):
    for o in sorted(obs.data, key=lambda o:o.name):
        print(" ", o.name, o.type.name, {f: round(float(obs.get(o,f)),3) for f in obs.type_features[o.type] if not f.startswith('color') and not f.startswith('v') and not f.startswith('omega')})
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print("seed", seed, "info", info)
    dump(obs)
print(env.max_steps)
env.close()
