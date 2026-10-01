from env_client import make_env


def dump(seed, count=None):
    env = make_env()
    opts = None if count is None else {"object_count": count}
    try:
        state, info = env.reset(seed=seed, options=opts)
    except TypeError:
        state, info = env.reset(seed=seed)
    print("seed", seed, "countarg", count, "info", info, "max", env.max_steps)
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        typ = obj.type.name if hasattr(obj.type, "name") else str(obj.type)
        feats = {
            "crv_robot": ("x", "y", "theta", "base_radius", "arm_joint", "arm_length", "vacuum", "gripper_height", "gripper_width"),
            "shelf": ("x", "y", "theta", "width", "height", "x1", "y1", "theta1", "width1", "height1"),
            "target_block": ("x", "y", "theta", "width", "height"),
        }[typ]
        vals = {f: round(float(state.get(obj, f)), 4) for f in feats}
        print(name, typ, vals)
    env.close()


for s in range(3):
    dump(s)
