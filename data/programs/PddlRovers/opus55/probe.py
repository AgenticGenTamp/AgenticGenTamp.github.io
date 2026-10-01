from env_client import make_env
env = make_env()
for seed in [0,1]:
    obs, info = env.reset(seed=seed)
    print("seed",seed, "info", info)
    for t in ["rover","lander","objective","sample","obstacle"]:
        for o in obs.get_objects(env.observation_space.get_type(t)) if hasattr(env.observation_space,'get_type') else []:
            print(o.name, [round(obs.get(o,f),3) for f in env.observation_space.type_features[env.observation_space.get_type(t)]] if hasattr(env.observation_space,'type_features') else None)
print(type(obs))
env.close()
