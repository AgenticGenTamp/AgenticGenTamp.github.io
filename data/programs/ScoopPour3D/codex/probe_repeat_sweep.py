"""Probe short recovery strokes after the production policy's first sweep."""

import argparse
import numpy as np

from approach import GeneratedApproach
from env_client import make_env


QFS = ["pos_arm_joint%d" % i for i in range(1, 8)]


def read(state, name, features):
    obj = state.get_object_from_name(name)
    return np.asarray([state.get(obj, f) for f in features], dtype=float)


def report(state, label, reward=0.0):
    names = [n for n in state.get_object_names() if n.startswith("cube_")]
    cubes = np.asarray([read(state, n, ("x", "y", "z")) for n in names])
    target = read(state, "bin_green_0", ("x", "y", "z"))
    scoop = read(state, "scoop_0", ("x", "y", "z"))
    base = read(state, "robot", ("pos_base_x", "pos_base_y", "pos_base_rot"))
    dist = np.linalg.norm(cubes[:, :2] - target[:2], axis=1)
    print(label, "r", round(reward, 3), "base", np.round(base, 3),
          "scoop", np.round(scoop, 3), "cube", np.round(cubes.mean(0), 3),
          "target", np.round(target, 3), "d", np.round(dist, 3),
          "near10", int(np.sum(dist < .10)), "near5", int(np.sum(dist < .05)))


def apply(env, state, count, values):
    total = 0.0
    for _ in range(count):
        action = np.zeros(11, dtype=np.float32)
        action[10] = 0.0
        for dim, value in values.items():
            action[dim] = value
        state, reward, done, trunc, _ = env.step(action)
        total += reward
        if done or trunc:
            break
    return state, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("reverse", "joint_only", "arm_reset", "regrasp", "lift_reverse", "retract_push"),
                        default="reverse")
    args = parser.parse_args()
    env = make_env()
    state, info = env.reset(seed=args.seed)
    policy_state = state
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    for _ in range(250):
        state, _, done, trunc, _ = env.step(policy.get_action(state))
        if done or trunc:
            break
    report(state, "start")

    if args.mode == "regrasp":
        # Release the collision-loaded grasp and recover the reset arm shape.
        for _ in range(45):
            action = np.zeros(11, np.float32); action[10] = 1.0
            q = read(state, "robot", QFS)
            action[3:10] = np.clip(.7 * (policy.home_joints - q), -.1, .1)
            state, _, _, _, _ = env.step(action)
        report(state, "released")
        initial_scoop = read(policy_state, "scoop_0", ("x", "y"))
        offset = policy.align_pose[:2] - initial_scoop
        for _ in range(30):
            action = np.zeros(11, np.float32); action[10] = 1.0
            base = read(state, "robot", ("pos_base_x", "pos_base_y", "pos_base_rot"))
            goal = np.r_[read(state, "scoop_0", ("x", "y")) + offset,
                         policy.align_pose[2]]
            error = goal - base
            error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
            action[:3] = np.clip(.55 * error, -.1, .1)
            q = read(state, "robot", QFS)
            action[3:10] = np.clip(.7 * (policy.reach_joints - q), -.1, .1)
            state, _, _, _, _ = env.step(action)
        report(state, "aligned")
        phases = [(30, {4: .10, 6: .06, 10: 1.0}),
                  (3, {10: 0.0}), (18, {4: -.10, 6: -.06}),
                  (30, {6: .10}), (35, {4: .06}),
                  (70, {1: .012, 5: -.10})]
        for i, (count, values) in enumerate(phases):
            state, total = apply(env, state, count, values)
            report(state, "rephase%d" % i, total)
        env.close()
        return

    if args.mode == "reverse":
        phases = [(30, {1: -.025, 5: .10}), (25, {1: .025, 5: -.10})]
    elif args.mode == "joint_only":
        # Reset/reverse joint 3 without translating the base into either bin,
        # then replay the original coordinated forward stroke.
        phases = [(30, {5: .10}), (45, {1: .012, 5: -.10})]
    elif args.mode == "arm_reset":
        phases = [(20, {4: -.08}), (30, {5: .10}),
                  (20, {4: .08}), (55, {1: .012, 5: -.10})]
    elif args.mode == "lift_reverse":
        phases = [(18, {4: -.08}), (35, {1: -.04, 5: .10}),
                  (18, {4: .08}), (40, {1: .025, 5: -.10})]
    else:
        phases = [(30, {6: -.10}), (20, {1: -.04}),
                  (30, {6: .10}), (50, {1: .025, 5: -.10})]

    for i, (count, values) in enumerate(phases):
        state, total = apply(env, state, count, values)
        report(state, "phase%d" % i, total)
    env.close()


if __name__ == "__main__":
    main()
