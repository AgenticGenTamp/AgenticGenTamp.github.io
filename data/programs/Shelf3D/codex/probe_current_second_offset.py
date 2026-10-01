"""Probe second-pick base calibration against the current policy."""

import sys

import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def run(seed, dx, dy):
    env = make_env()
    state, info = env.reset(seed=seed, options={"object_count": 2})
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    high = [0.0, 0.0]
    try:
        for step in range(1000):
            action = policy.get_action(state)
            if policy.index == 1 and policy.stage <= 5 and policy.pick_xy is not None:
                robot = state.get_object_from_name("robot")
                bx = float(state.get(robot, "pos_base_x"))
                by = float(state.get(robot, "pos_base_y"))
                tx = policy.pick_xy[0] - .609 + dx
                ty = policy.pick_xy[1] - .054 + dy
                if policy.stage == 0:
                    tx -= .195
                action[0] = np.clip((tx - bx) / .87, -.1, .1)
                action[1] = np.clip((ty - by) / .87, -.1, .1)
            state, reward, term, trunc, _ = env.step(action)
            for i, name in enumerate(policy.cube_names):
                obj = state.get_object_from_name(name)
                high[i] = max(high[i], float(state.get(obj, "z")))
            if term or trunc:
                break
        print("seed", seed, "offset", dx, dy, "high", high,
              "policy", policy.index, policy.stage, policy.retry,
              "done", term, trunc, "reward", reward)
    finally:
        env.close()


if __name__ == "__main__":
    run(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
