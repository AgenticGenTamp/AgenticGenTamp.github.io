"""Compact reset-state probe for DynPushPullHook2DEnv."""

from collections import Counter
from env_client import make_env


FEATURES = {
    "robot": ("x", "y", "theta", "base_radius", "arm_joint", "arm_length",
              "gripper_base_width", "gripper_base_height", "finger_gap",
              "finger_height", "finger_width"),
    "hook": ("x", "y", "theta", "width", "length_side1", "length_side2", "mass"),
    "target_block": ("x", "y", "theta", "width", "height", "mass"),
}


def vals(state, obj, features):
    return tuple(round(float(state.get(obj, f)), 4) for f in features)


def summarize(state):
    names = state.get_object_names()
    by_type = Counter(state.get_object_from_name(n).type.name for n in names)
    result = {"counts": dict(sorted(by_type.items()))}
    for name, features in FEATURES.items():
        obj = state.get_object_from_name(name)
        result[name] = vals(state, obj, features)
    # Names are the stable way to distinguish the generic rectangle obstacles.
    obs = sorted(
        (state.get_object_from_name(n) for n in names if n.startswith("obstruction")),
        key=lambda o: o.name,
    )
    result["obstructions"] = [vals(state, o, ("x", "y", "theta", "width", "height", "mass")) for o in obs]
    return result


def main():
    env = make_env()
    try:
        print("max_steps", env.max_steps)
        print("action", env.action_space.shape, env.action_space.low.tolist(), env.action_space.high.tolist())
        for seed in range(16):
            state, info = env.reset(seed=seed)
            print("default", seed, summarize(state), "info", info)
        for count in (0, 1, 2, 3, 4, 6, 8, 12):
            for seed in (0, 1, 7):
                try:
                    state, info = env.reset(seed=seed, options={"object_count": count})
                    print("pinned", count, seed, summarize(state), "info", info)
                except Exception as exc:
                    print("pinned_error", count, seed, type(exc).__name__, str(exc))
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
