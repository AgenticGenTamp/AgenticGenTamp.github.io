from env_client import make_env


def value(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


env = make_env()
try:
    for count in (1, 2, 3, 4, 6, 8):
        print("COUNT", count)
        for seed in range(3):
            state, info = env.reset(seed=seed, options={"object_count": count})
            cubes = sorted((n for n in state.get_object_names() if n.startswith("cube")),
                           key=lambda n: int(n[4:]))
            xyz = [(n, round(value(state, n, "x"), 3),
                    round(value(state, n, "y"), 3),
                    round(value(state, n, "z"), 3)) for n in cubes]
            robot = tuple(round(value(state, "robot", f), 3) for f in
                          ("pos_base_x", "pos_base_y", "pos_base_rot"))
            print(seed, "robot", robot, "cubes", xyz)
    counts = []
    for seed in range(30):
        _, info = env.reset(seed=seed)
        counts.append(info["object_count"])
    print("DEFAULT_COUNTS", counts)
finally:
    env.close()
