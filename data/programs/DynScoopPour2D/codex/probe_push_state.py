from env_client import make_env


def val(s, o, f):
    return float(s.get(o, f))


for seed in range(3):
    env = make_env()
    state, info = env.reset(seed=seed)
    print("seed", seed, "max", env.max_steps, "info", info)
    for name in state.get_object_names():
        o = state.get_object_from_name(name)
        fs = []
        for f in ("x", "y", "theta", "radius", "size", "width", "length_side1", "length_side2", "arm_length", "finger_gap", "held"):
            try:
                fs.append(f"{f}={val(state,o,f):.3f}")
            except Exception:
                pass
        print(name, " ".join(fs))
    env.close()
