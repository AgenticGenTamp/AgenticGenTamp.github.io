from env_client import make_env
import sys
env = make_env()
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    obs, info = env.reset(seed=seed)
    print("seed", seed, info)
    for n in sorted(obs.get_object_names(), key=str):
        o = obs.get_object_from_name(n) if isinstance(n,str) else n
        t = o.type
        feats = env.observation_space.type_features[t] if hasattr(env.observation_space,'type_features') else None
        print(" ", o.name, t.name if hasattr(t,'name') else t, {f: round(float(obs.get(o,f)),3) for f in feats})
