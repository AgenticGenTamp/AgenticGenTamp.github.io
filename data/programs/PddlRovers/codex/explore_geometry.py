from env_client import make_env


def dump(seed):
    env = make_env()
    state, info = env.reset(seed=seed)
    print("SEED", seed, "max", env.max_steps, "info", info)
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        vals = [(f, float(state.get(obj, f))) for f in state.type_features[obj.type]]
        print(name, obj.type.name, vals)
    env.close()


for seed in range(6):
    dump(seed)
