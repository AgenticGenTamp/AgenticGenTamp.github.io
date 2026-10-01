"""Compact black-box probes for initial states and action semantics."""

import sys

import numpy as np

from env_client import make_env


def val(state, obj, feature):
    return float(state.get(obj, feature))


def objects(state, typ):
    return list(state.get_objects(OBS_SPACE.get_type(typ)))


def snapshot(state):
    result = {}
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        # Feature availability is easiest to infer from known object roles here.
        fields = ["x", "y", "theta"]
        result[name] = {field: val(state, obj, field) for field in fields}
    return result


def initial_probe():
    counts = []
    extrema = {"small_x": [], "small_y": [], "robot": [], "hook": []}
    for seed in range(30):
        env = make_env()
        state, info = env.reset(seed=seed)
        small = objects(state, "small_circle") + objects(state, "small_square")
        counts.append(len(small))
        extrema["small_x"].extend(val(state, obj, "x") for obj in small)
        extrema["small_y"].extend(val(state, obj, "y") for obj in small)
        robot = objects(state, "kin_robot")[0]
        hook = objects(state, "hook")[0]
        extrema["robot"].append((val(state, robot, "x"), val(state, robot, "y"), val(state, robot, "theta")))
        extrema["hook"].append((val(state, hook, "x"), val(state, hook, "y"), val(state, hook, "theta")))
        if seed < 4:
            print("INIT", seed, "n", len(small), "info", info, snapshot(state))
            print("ROBOT_GEOM", seed, {f: val(state, robot, f) for f in ["base_radius", "arm_joint", "arm_length", "gripper_base_width", "gripper_base_height", "finger_gap", "finger_height", "finger_width"]})
            print("HOOK_GEOM", seed, {f: val(state, hook, f) for f in ["width", "length_side1", "length_side2", "mass", "held", "static"]})
            for obj in small:
                kind = "circle" if obj in objects(state, "small_circle") else "square"
                extra = "radius" if kind == "circle" else "size"
                print("SMALL", seed, obj.name, kind, extra, val(state, obj, extra), "mass", val(state, obj, "mass"))
        env.close()
    print("COUNTS", counts, "unique", sorted(set(counts)))
    for key, seq in extrema.items():
        arr = np.asarray(seq)
        print("RANGE", key, "min", arr.min(axis=0), "max", arr.max(axis=0))


def action_probe():
    actions = {
        "dx+": [0.03, 0, 0, 0, 0],
        "dy+": [0, 0.03, 0, 0, 0],
        "rot+": [0, 0, 0.098, 0, 0],
        "arm+": [0, 0, 0, 0.08, 0],
        "grip-": [0, 0, 0, 0, -0.015],
        "grip+": [0, 0, 0, 0, 0.015],
    }
    robot_fields = ["x", "y", "theta", "arm_joint", "arm_length", "finger_gap"]
    for label, action in actions.items():
        if len(sys.argv) > 2 and label != sys.argv[2]:
            continue
        env = make_env()
        state, _ = env.reset(seed=0)
        robot = objects(state, "kin_robot")[0]
        before = np.array([val(state, robot, f) for f in robot_fields])
        state, reward, term, trunc, info = env.step(np.array(action, dtype=float))
        robot = objects(state, "kin_robot")[0]
        after = np.array([val(state, robot, f) for f in robot_fields])
        print("ACTION", label, "delta", after - before, "post", after, reward, term, trunc, info)
        env.close()

    # Sustained motions show bounds and whether input is direct delta or velocity-like.
    repeated = [
        ("dx+", [0.03, 0, 0, 0, 0]),
        ("dx-", [-0.03, 0, 0, 0, 0]),
        ("dy+", [0, 0.03, 0, 0, 0]),
        ("dy-", [0, -0.03, 0, 0, 0]),
        ("arm+", [0, 0, 0, 0.08, 0]),
        ("arm-", [0, 0, 0, -0.08, 0]),
        ("grip-", [0, 0, 0, 0, -0.015]),
        ("grip+", [0, 0, 0, 0, 0.015]),
    ]
    for label, action in repeated:
        if len(sys.argv) > 2 and label != sys.argv[2]:
            continue
        env = make_env()
        state, _ = env.reset(seed=0)
        for step in range(1, 41):
            state, reward, term, trunc, info = env.step(np.array(action, dtype=float))
            if step in (1, 2, 5, 10, 20, 30, 40) or term or trunc:
                robot = objects(state, "kin_robot")[0]
                print("REPEAT", label, step, {f: val(state, robot, f) for f in robot_fields}, reward, term, trunc)
            if term or trunc:
                break
        env.close()


if __name__ == "__main__":
    env = make_env()
    OBS_SPACE = env.observation_space
    print("SPACE", env.max_steps, env.action_space.low, env.action_space.high, env.observation_space.types)
    env.close()
    if len(sys.argv) == 1 or sys.argv[1] == "initial":
        initial_probe()
    if len(sys.argv) == 1 or sys.argv[1] == "actions":
        action_probe()
