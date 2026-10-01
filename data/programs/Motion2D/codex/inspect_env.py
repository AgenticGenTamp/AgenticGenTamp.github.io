from env_client import make_env

for seed in range(5):
    env = make_env()
    s, info = env.reset(seed=seed)
    print("seed", seed, "info", info, "max", env.max_steps)
    for name in s.get_object_names():
        o = s.get_object_from_name(name)
        typ = getattr(o, "type", None)
        tn = typ.name if hasattr(typ, 'name') else str(typ)
        feats = (["x", "y", "theta", "base_radius", "arm_joint", "arm_length", "vacuum", "gripper_height", "gripper_width"]
                 if tn == "crv_robot" else
                 ["x", "y", "theta", "static", "color_r", "color_g", "color_b", "z_order", "width", "height"])
        print(" ", name, tn, {f: round(float(s.get(o, f)), 4) for f in feats})
    env.close()
