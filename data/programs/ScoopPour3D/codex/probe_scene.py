"""Small black-box probes for ScoopPour3DEnv (never imported by approach.py)."""

import argparse
import math

import numpy as np

from env_client import make_env


def val(state, obj, feature):
    return state.get(obj, feature)


def movable_rows(env, state):
    typ = env.observation_space.get_type("mujoco_movable_object")
    rows = []
    for obj in state.get_objects(typ):
        rows.append(
            (
                obj.name,
                *(val(state, obj, f) for f in ("x", "y", "z")),
                *(val(state, obj, f) for f in ("bb_x", "bb_y", "bb_z")),
            )
        )
    return rows


def robot_row(env, state):
    typ = env.observation_space.get_type("mujoco_tidybot_robot")
    obj = state.get_objects(typ)[0]
    fs = ["pos_base_x", "pos_base_y", "pos_base_rot"]
    fs += [f"pos_arm_joint{i}" for i in range(1, 8)]
    fs += ["pos_gripper"]
    return np.array([val(state, obj, f) for f in fs])


def print_init(seeds, counts):
    for count in counts:
        for seed in seeds:
            env = make_env()
            options = None if count < 0 else {"object_count": count}
            state, info = env.reset(seed=seed, options=options)
            print(f"INIT seed={seed} requested_count={count} info={info}")
            for row in movable_rows(env, state):
                print("  %-13s xyz=(% .4f,% .4f,% .4f) bb=(%.4f,%.4f,%.4f)" % row)
            print("  robot", np.array2string(robot_row(env, state), precision=4))
            env.close()


def step_probe(seed, count, repeats):
    actions = [("zero_g0", np.zeros(11)), ("zero_g1", np.r_[np.zeros(10), 1.0])]
    for i in range(10):
        for sign in (-1.0, 1.0):
            action = np.zeros(11)
            action[i] = 0.1 * sign
            action[10] = 1.0
            actions.append((f"a{i}_{sign:+.0f}", action))
    for label, action in actions:
        env = make_env()
        state0, _ = env.reset(seed=seed, options={"object_count": count})
        r0 = robot_row(env, state0)
        xyz0 = {r[0]: np.array(r[1:4]) for r in movable_rows(env, state0)}
        rewards = []
        done = None
        state = state0
        for k in range(repeats):
            state, reward, term, trunc, info = env.step(action)
            rewards.append(reward)
            if term or trunc:
                done = (k + 1, term, trunc, info)
                break
        dr = robot_row(env, state) - r0
        changes = []
        for row in movable_rows(env, state):
            d = np.array(row[1:4]) - xyz0[row[0]]
            if np.linalg.norm(d) > 1e-4:
                changes.append(f"{row[0]}:{np.round(d,4)}")
        print(label, "rewards", np.round(rewards, 4), "drobot", np.round(dr, 4),
              "moved", changes, "done", done)
        env.close()


def rollout(seed, count, mode, steps, direction=1.0):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": count})
    initial = {r[0]: np.array(r[1:4]) for r in movable_rows(env, state)}
    total = 0.0
    rng = np.random.default_rng(seed + 1000)
    for k in range(steps):
        if mode == "zero":
            action = np.zeros(11)
        elif mode == "random":
            action = rng.uniform(env.action_space.low, env.action_space.high)
        elif mode.startswith("axis"):
            action = np.zeros(11)
            suffix = mode[4:]
            closed = suffix.endswith("c")
            if closed:
                suffix = suffix[:-1]
            action[int(suffix)] = 0.1 * direction
            action[10] = 0.0 if closed else 1.0
        state, reward, term, trunc, info = env.step(action)
        total += reward
        if k < 5 or reward != -0.01 or term or trunc or (k + 1) % 25 == 0:
            print(k + 1, "reward", reward, "total", round(total, 4), "done", term, trunc)
        if term or trunc:
            break
    print("final robot", np.round(robot_row(env, state), 4))
    for row in movable_rows(env, state):
        d = np.array(row[1:4]) - initial[row[0]]
        print(" ", row[0], "xyz", np.round(row[1:4], 4), "delta", np.round(d, 4))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["init", "step", "rollout"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--counts", default="0,1,3,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--policy", default="zero")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--direction", type=float, default=1.0)
    args = parser.parse_args()
    if args.mode == "init":
        print_init([int(x) for x in args.seeds.split(",")],
                   [int(x) for x in args.counts.split(",")])
    elif args.mode == "step":
        step_probe(args.seed, args.count, args.repeats)
    else:
        rollout(args.seed, args.count, args.policy, args.steps, args.direction)
