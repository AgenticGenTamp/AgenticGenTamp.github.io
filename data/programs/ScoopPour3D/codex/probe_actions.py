"""Empirically probe ScoopPour3DEnv action coordinates on matched seeds."""

import argparse

import numpy as np

from env_client import make_env


ROBOT_FEATURES = [
    "pos_base_x", "pos_base_y", "pos_base_rot",
    "pos_arm_joint1", "pos_arm_joint2", "pos_arm_joint3", "pos_arm_joint4",
    "pos_arm_joint5", "pos_arm_joint6", "pos_arm_joint7", "pos_gripper",
]
MOVABLE_FEATURES = ["x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz"]


def values(state, obj, features):
    return np.asarray([state.get(obj, f) for f in features], dtype=float)


def snapshot(state):
    names = state.get_object_names()
    robot = state.get_object_from_name("robot")
    result = {"robot": values(state, robot, ROBOT_FEATURES)}
    for name in names:
        if name.startswith(("cube_", "bin_", "scoop_")):
            result[name] = values(state, state.get_object_from_name(name), MOVABLE_FEATURES)
    return result


def run(env, seed, dim, magnitude, steps):
    state, info = env.reset(seed=seed)
    before = snapshot(state)
    action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    action[dim] = magnitude
    rewards = []
    terminated = False
    for _ in range(steps):
        state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    after = snapshot(state)
    return before, after, np.asarray(rewards), terminated


def fmt(a):
    return np.array2string(a, precision=5, suppress_small=True, floatmode="fixed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--magnitude", type=float, default=0.1)
    parser.add_argument("--dims", type=int, nargs="*", default=list(range(11)))
    parser.add_argument("--verbose-initial", action="store_true")
    args = parser.parse_args()

    env = make_env()
    state, info = env.reset(seed=args.seed)
    base = snapshot(state)
    print("action range", fmt(env.action_space.low), fmt(env.action_space.high))
    print("objects", state.get_object_names())
    print("initial robot", fmt(base["robot"]), "tracked_objects", len(base) - 1)
    if args.verbose_initial:
        for name, val in base.items():
            if name != "robot":
                print("initial", name, fmt(val))

    for dim in args.dims:
        magnitudes = (0.0, 1.0) if dim == 10 else (-args.magnitude, args.magnitude)
        for magnitude in magnitudes:
            before, after, rewards, terminated = run(
                env, args.seed, dim, magnitude, args.steps
            )
            print(
                "probe", dim, magnitude, "robot_delta", fmt(after["robot"] - before["robot"]),
                "rewards", fmt(rewards), "terminated", terminated,
            )
            for name in before:
                if name != "robot":
                    delta = after[name] - before[name]
                    if np.max(np.abs(delta)) > 1e-5:
                        print("  object_delta", name, fmt(delta))
    env.close()


if __name__ == "__main__":
    main()
