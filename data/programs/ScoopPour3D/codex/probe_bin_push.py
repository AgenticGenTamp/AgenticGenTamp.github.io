"""Probe whether a base-driven arm sweep can push a bin over the cubes."""

import argparse

import numpy as np

from env_client import make_env


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, f) for f in ("x", "y", "z")], dtype=float)


def robot_pose(state):
    obj = state.get_object_from_name("robot")
    return np.array([
        state.get(obj, "pos_base_x"), state.get(obj, "pos_base_y"),
        state.get(obj, "pos_base_rot"),
    ], dtype=float)


def cube_summary(state):
    ps = np.array([
        xyz(state, n) for n in state.get_object_names() if n.startswith("cube_")
    ])
    return len(ps), ps.min(axis=0), ps.mean(axis=0), ps.max(axis=0)


def drive(env, state, target, max_steps=30):
    """Drive base x/y/yaw to target with saturated proportional velocity."""
    total = 0.0
    for i in range(max_steps):
        pose = robot_pose(state)
        error = np.asarray(target) - pose
        error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
        if np.max(np.abs(error)) < 0.025:
            return state, total, False
        action = np.zeros(11, dtype=np.float32)
        action[:3] = np.clip(error * np.array([1.2, 1.2, 1.5]), -0.1, 0.1)
        action[10] = 0.0
        state, reward, term, trunc, _ = env.step(action)
        total += reward
        if i % 3 == 0 or reward != -1.0:
            print(" step", i, "base", np.round(robot_pose(state), 3),
                  "bins", np.round(xyz(state, "bin_green_0"), 3),
                  np.round(xyz(state, "bin_yellow_0"), 3), "r", reward)
        if term or trunc:
            return state, total, True
    return state, total, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=("green_to_cubes", "yellow_to_green"),
                    default="green_to_cubes")
    args = ap.parse_args()
    env = make_env()
    state, _ = env.reset(seed=args.seed)
    yaw = robot_pose(state)[2]
    print("initial base", np.round(robot_pose(state), 3), "bins",
          np.round(xyz(state, "bin_green_0"), 3),
          np.round(xyz(state, "bin_yellow_0"), 3), "cubes", cube_summary(state))

    # The initial arm endpoint is approximately 0.42 m in front of the base and
    # at bin height. Approach a bin from its outside, then sweep through it.
    if args.mode == "green_to_cubes":
        waypoints = [(-0.21, 0.66, yaw), (0.08, 0.66, yaw),
                     (0.08, 0.31, yaw), (0.08, -0.28, yaw)]
    else:
        waypoints = [(-0.21, -0.66, yaw), (0.08, -0.66, yaw),
                     (0.08, -0.31, yaw), (0.08, 0.28, yaw)]
    total = 0.0
    for target in waypoints:
        print("target", target)
        state, reward, done = drive(env, state, target)
        total += reward
        print(" reached", np.round(robot_pose(state), 3), "bins",
              np.round(xyz(state, "bin_green_0"), 3),
              np.round(xyz(state, "bin_yellow_0"), 3), "cubes", cube_summary(state))
        if done:
            break
    print("final total", total, "done", done)
    env.close()


if __name__ == "__main__":
    main()
