"""Probe live cube-relative base servo during the second alignment stage."""

import argparse
import math
import numpy as np

from env_client import make_env
from approach import GeneratedApproach


def cube_xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, key) for key in ("x", "y", "z")])


def run(seed, rel_x, rel_y):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 2})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    best_second_z = 0.0
    second_lifted = False
    terminated = False
    final_reward = None
    for step in range(env.max_steps):
        before_index, before_stage = policy.index, policy.stage
        action = policy.get_action(state)
        if before_index == 1 and before_stage == 2:
            cube = cube_xyz(state, policy.cube_names[1])
            robot = state.get_object_from_name("robot")
            base = np.array([state.get(robot, "pos_base_x"),
                             state.get(robot, "pos_base_y")])
            target = cube[:2] - np.array([rel_x, rel_y])
            action[:2] = np.clip((target - base) / .87, -.1, .1)
        state, final_reward, terminated, truncated, _ = env.step(action)
        if len(policy.cube_names) > 1:
            z = cube_xyz(state, policy.cube_names[1])[2]
            best_second_z = max(best_second_z, float(z))
            second_lifted = second_lifted or z > .08
        if terminated or truncated:
            break
    env.close()
    return step + 1, second_lifted, best_second_z, terminated, final_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=.727)
    parser.add_argument("--y", type=float, default=.07)
    parser.add_argument("seeds", nargs="*", type=int,
                        default=[0, 1, 2, 4])
    args = parser.parse_args()
    for seed in args.seeds:
        print(seed, run(seed, args.x, args.y), flush=True)
