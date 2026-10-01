from env_client import make_env

for seed in range(5):
    env = make_env()
    obs, info = env.reset(seed=seed)
    print("SEED", seed, "max", env.max_steps, "info", info)
    for name in obs.get_object_names():
        obj = obs.get_object_from_name(name)
        typ = getattr(obj, "type", None)
        vals = {}
        for f in ("x", "y", "theta", "width", "height", "mass", "held", "base_radius", "arm_joint", "arm_length", "finger_gap", "finger_height"):
            try: vals[f] = round(float(obs.get(obj, f)), 4)
            except Exception: pass
        print(name, typ, vals)
    env.close()
