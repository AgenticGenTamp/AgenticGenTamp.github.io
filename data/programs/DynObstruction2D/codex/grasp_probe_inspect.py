from env_client import make_env


def dump(seed):
    env = make_env()
    state, info = env.reset(seed=seed)
    print("seed", seed, "max", env.max_steps, "info", info)
    for typ in env.observation_space.types:
        for obj in state.get_objects(typ):
            vals = {f: state.get(obj, f) for f in env.observation_space.type_features[typ]}
            print(typ, obj, vals)
    env.close()


for seed in range(4):
    dump(seed)
