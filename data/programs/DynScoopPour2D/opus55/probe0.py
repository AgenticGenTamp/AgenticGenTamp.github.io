from env_client import make_env
env = make_env()
for seed in range(3):
    obs, info = env.reset(seed=seed)
    print("seed", seed, info)
    for n in obs.get_object_names():
        o = obs.get_object_from_name(n)
        t = o.type if hasattr(o,'type') else None
        feats = env.observation_space.get_type(t.name if hasattr(t,'name') else t) if t else None
        print(n, t, {f: round(obs.get(o,f),3) for f in (env.observation_space.type_features[t] if t in env.observation_space.type_features else [])})
env.close()
