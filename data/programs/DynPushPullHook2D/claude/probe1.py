from env_client import make_env
env = make_env()
tf = {t.name: list(f) for t,f in env.observation_space.type_features.items()}
for seed in [0,42]:
    obs, info = env.reset(seed=seed)
    print("=== seed", seed, info)
    for name in sorted(obs.get_object_names()):
        o = obs.get_object_from_name(name)
        print(name, o.type.name, {f: round(float(obs.get(o,f)),4) for f in tf[o.type.name]})
env.close()
