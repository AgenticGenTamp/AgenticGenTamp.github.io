from env_client import make_env


def val(s, o, f):
    return float(s.get(o, f))


for seed in range(5):
    env = make_env()
    obs, info = env.reset(seed=seed)
    print("SEED", seed, "max", env.max_steps, "info", info)
    for typ in env.observation_space.types:
        objs = obs.get_objects(typ)
        fs = env.observation_space.type_features[typ]
        print(typ.name if hasattr(typ, "name") else typ, len(objs), fs)
        for o in objs:
            print(" ", o.name, [round(val(obs, o, f), 3) for f in fs])
    env.close()
