from env_client import make_env


def dump(seed):
    env = make_env()
    state, info = env.reset(seed=seed)
    print("SEED", seed, "max_steps", env.max_steps, "info", info)
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        typ = getattr(obj, "type", getattr(obj, "type_name", "?"))
        vals = {}
        for feat in ("x", "y", "theta", "radius", "size", "width", "length_side1", "length_side2", "base_radius", "arm_joint", "arm_length", "finger_gap", "finger_height", "finger_width", "held"):
            try:
                vals[feat] = round(float(state.get(obj, feat)), 4)
            except Exception:
                pass
        print(name, typ, vals)
    env.close()


for s in range(3):
    dump(s)
