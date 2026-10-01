"""Compact reset-state survey for DynObstruction2DEnv."""

from collections import Counter, defaultdict

from env_client import make_env


FEATURES = {
    "target_block": ("x", "y", "theta", "width", "height", "mass"),
    "target_surface": ("x", "y", "theta", "width", "height"),
    "kin_robot": (
        "x", "y", "theta", "base_radius", "arm_joint", "arm_length",
        "gripper_base_width", "gripper_base_height", "finger_gap",
        "finger_height", "finger_width",
    ),
    "dyn_rectangle": ("x", "y", "theta", "width", "height", "mass"),
}


def one_state(env, seed, options=None):
    state, info = env.reset(seed=seed, options=options)
    rows = {}
    for type_name, features in FEATURES.items():
        typ = env.observation_space.get_type(type_name)
        rows[type_name] = [
            (obj.name, {f: state.get(obj, f) for f in features})
            for obj in state.get_objects(typ)
        ]
    return rows, info


def main():
    env = make_env()
    print("max_steps", env.max_steps)
    print("action", env.action_space.shape, env.action_space.low, env.action_space.high)
    count_hist = Counter()
    ranges = defaultdict(lambda: [float("inf"), -float("inf")])
    examples = {}
    for seed in range(40):
        rows, _ = one_state(env, seed)
        # dyn_rectangle includes target_block due to subtype semantics.
        obs = [r for r in rows["dyn_rectangle"] if r[0].startswith("obstruction")]
        count_hist[len(obs)] += 1
        examples.setdefault(len(obs), (seed, rows))
        for type_name, typed_rows in rows.items():
            for name, vals in typed_rows:
                if type_name == "dyn_rectangle" and name == "target_block":
                    continue
                for f, value in vals.items():
                    key = (type_name, f)
                    ranges[key][0] = min(ranges[key][0], value)
                    ranges[key][1] = max(ranges[key][1], value)
    print("count_hist", dict(sorted(count_hist.items())))
    for key in sorted(ranges):
        print("range", key, tuple(round(x, 6) for x in ranges[key]))
    for count, (seed, rows) in sorted(examples.items()):
        print("example", count, "seed", seed)
        for type_name in ("kin_robot", "target_block", "target_surface", "dyn_rectangle"):
            typed_rows = rows[type_name]
            if type_name == "dyn_rectangle":
                typed_rows = [r for r in typed_rows if r[0].startswith("obstruction")]
            print(type_name, typed_rows)

    # Probe likely reset option spelling once; report support without depending on it.
    for options in ({"object_count": 0}, {"object_count": 5}):
        try:
            rows, _ = one_state(env, 123, options)
            obs = [r for r in rows["dyn_rectangle"] if r[0].startswith("obstruction")]
            print("options", options, "observed_obstructions", len(obs))
        except Exception as exc:
            print("options", options, "error", type(exc).__name__, str(exc))
    env.close()


if __name__ == "__main__":
    main()
