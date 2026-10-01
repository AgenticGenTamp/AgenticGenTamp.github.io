"""Empirical control/kinematics probes for Transport3DEnv."""

import argparse
import json

import numpy as np

from env_client import make_env


def snapshot(state):
    out = {}
    for name in state.get_object_names():
        obj = state.get_object_from_name(name)
        if name == "robot":
            fs = [
                "pos_base_x", "pos_base_y", "pos_base_rot",
                "joint_1", "joint_2", "joint_3", "joint_4",
                "joint_5", "joint_6", "joint_7", "finger_state",
                "grasp_active", "grasp_tf_x", "grasp_tf_y", "grasp_tf_z",
                "grasp_tf_qx", "grasp_tf_qy", "grasp_tf_qz", "grasp_tf_qw",
            ]
        else:
            fs = [
                "pose_x", "pose_y", "pose_z", "pose_qx", "pose_qy",
                "pose_qz", "pose_qw", "grasp_active", "object_type",
                "half_extent_x", "half_extent_y", "half_extent_z",
            ]
        vals = {}
        for feature in fs:
            try:
                vals[feature] = float(state.get(obj, feature))
            except Exception:
                pass
        out[name] = vals
    return out


def delta(before, after, name="robot"):
    return {
        k: round(after[name][k] - v, 6)
        for k, v in before[name].items()
        if k in after[name] and abs(after[name][k] - v) > 1e-7
    }


def axis_probe(seed):
    for axis in range(11):
        env = make_env()
        state, _ = env.reset(seed=seed)
        before = snapshot(state)
        action = np.zeros(11, dtype=np.float32)
        action[axis] = 0.2 if axis < 10 else 1.0
        state, reward, term, trunc, info = env.step(action)
        after = snapshot(state)
        print(json.dumps({"axis": axis, "delta": delta(before, after),
                          "reward": reward, "term": term}))
        env.close()


def trajectory(seed, axis, value, steps):
    env = make_env()
    state, _ = env.reset(seed=seed)
    print(json.dumps({"step": 0, "state": snapshot(state)}))
    action = np.zeros(11, dtype=np.float32)
    action[axis] = value
    for step in range(1, steps + 1):
        state, reward, term, trunc, info = env.step(action)
        print(json.dumps({"step": step, "state": snapshot(state),
                          "reward": reward, "term": term, "trunc": trunc}))
        if term or trunc:
            break
    env.close()


def move_base(env, state, target_x, target_y, rotation=0.0):
    robot = state.get_object_from_name("robot")
    for _ in range(20):
        dx = target_x - float(state.get(robot, "pos_base_x"))
        dy = target_y - float(state.get(robot, "pos_base_y"))
        dr = rotation - float(state.get(robot, "pos_base_rot"))
        if max(abs(dx), abs(dy), abs(dr)) < 1e-4:
            break
        action = np.zeros(11, dtype=np.float32)
        action[:3] = np.clip([dx, dy, dr], -0.2, 0.2)
        state, _, _, _, _ = env.step(action)
    return state


def set_joints(env, state, values):
    robot = state.get_object_from_name("robot")
    for _ in range(40):
        current = np.array([float(state.get(robot, "joint_%d" % (i + 1)))
                            for i in range(7)])
        diff = np.asarray(values) - current
        if np.max(np.abs(diff)) < 1e-4:
            break
        action = np.zeros(11, dtype=np.float32)
        action[3:10] = np.clip(diff, -0.2, 0.2)
        state, _, _, _, _ = env.step(action)
    return state


def grasp_grid(seed, spacing, radius, target):
    # Probe default arm posture with the base at offsets around a target.
    points = np.arange(-radius, radius + spacing / 2, spacing)
    for off_x in points:
        for off_y in points:
            env = make_env()
            state, _ = env.reset(seed=seed)
            initial = snapshot(state)
            cube = initial[target]
            bx, by = cube["pose_x"] + off_x, cube["pose_y"] + off_y
            state = move_base(env, state, bx, by)
            action = np.zeros(11, dtype=np.float32)
            action[10] = 1.0
            state, _, _, _, _ = env.step(action)
            action[10] = -1.0
            for _ in range(2):
                state, _, _, _, _ = env.step(action)
            now = snapshot(state)
            active = now["robot"]["grasp_active"] or now[target]["grasp_active"]
            if active:
                print(json.dumps({"offset": [off_x, off_y], "state": now}))
            env.close()


def joint_search(seed, count):
    rng = np.random.default_rng(seed + 9182)
    default = np.array([0.0, -0.35, -np.pi, -2.5, 0.0, -0.87, np.pi / 2])
    configs = [np.array([0.0, -1.6743, -3.8876, -1.5737,
                         0.1635, -1.7428, 0.0253])]
    for q4 in np.linspace(-3.0, 0.0, 7):
        q = default.copy()
        q[3] = q4
        configs.append(q)
    for q3 in np.linspace(-2.75, -1.5, 6):
        q = default.copy()
        q[2] = q3
        configs.append(q)
    for _ in range(count):
        configs.append(default + rng.uniform(-1.5, 1.5, 7))
    points = np.arange(-0.75, 0.751, 0.25)
    for index, joints in enumerate(configs):
        env = make_env()
        state, _ = env.reset(seed=seed)
        target_state = snapshot(state)["box0"]
        state = set_joints(env, state, joints)
        for off_x in points:
            for off_y in points:
                state = move_base(env, state, target_state["pose_x"] + off_x,
                                  target_state["pose_y"] + off_y)
                action = np.zeros(11, dtype=np.float32)
                action[10] = 1.0
                state, _, _, _, _ = env.step(action)
                action[10] = -1.0
                state, _, _, _, _ = env.step(action)
                now = snapshot(state)
                if now["robot"]["grasp_active"]:
                    print(json.dumps({"config_index": index,
                                      "joints": joints.tolist(),
                                      "offset": [off_x, off_y],
                                      "state": now}))
                    env.close()
                    return
        env.close()
        print(json.dumps({"searched_config": index, "joints": joints.tolist()}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["initial", "axes", "trajectory", "graspgrid", "jointsearch"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--value", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--spacing", type=float, default=0.2)
    parser.add_argument("--radius", type=float, default=0.6)
    parser.add_argument("--target", default="cube1")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.mode == "initial":
        env = make_env()
        state, info = env.reset(seed=args.seed)
        print(json.dumps({"state": snapshot(state), "info": info}, indent=2))
        env.close()
    elif args.mode == "axes":
        axis_probe(args.seed)
    elif args.mode == "trajectory":
        trajectory(args.seed, args.axis, args.value, args.steps)
    elif args.mode == "graspgrid":
        grasp_grid(args.seed, args.spacing, args.radius, args.target)
    else:
        joint_search(args.seed, args.count)


if __name__ == "__main__":
    main()
