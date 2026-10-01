from env_client import make_env


def val(s, o, f):
    return float(s.get(o, f))


for seed in range(3):
    env = make_env()
    s, info = env.reset(seed=seed)
    print("SEED", seed, "info", info, "names", s.get_object_names())
    for name in s.get_object_names():
        o = s.get_object_from_name(name)
        typ = getattr(o, "type", None)
        print(name, typ, o)
        if name == "robot":
            fs = env.observation_space.type_features[typ]
        else:
            fs = env.observation_space.type_features[typ]
        print({f: val(s, o, f) for f in fs})
    print("action", env.action_space.low, env.action_space.high, "max", env.max_steps)
    env.close()
