import sys
from env_client import make_env
env = make_env()
for seed in [0,1,2]:
    obs, info = env.reset(seed=seed)
    print("seed",seed, "info", info)
    for n in sorted(obs.get_object_names(), key=str):
        o = obs.get_object_from_name(str(n)) if not hasattr(n,'name') else n
        t = o.type if hasattr(o,'type') else None
        feats = env.observation_space.type_features[t] if t is not None else []
        print(" ", o.name if hasattr(o,'name') else o, t.name if hasattr(t,'name') else t, {f: round(float(obs.get(o,f)),3) for f in feats})
print(env.max_steps)
env.close()
