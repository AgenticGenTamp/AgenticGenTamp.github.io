"""Search around the current pickup pose for an obstructed/attached grasp."""

import itertools
import math
import numpy as np
from env_client import make_env


def value(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


def robot_xy(state):
    return np.array([value(state, "robot", "pos_base_x"),
                     value(state, "robot", "pos_base_y")])


def wiper_xy(state):
    return np.array([value(state, "wiper_0", "x"),
                     value(state, "wiper_0", "y")])


def angle_error(target, actual):
    return (target - actual + math.pi) % (2 * math.pi) - math.pi


def run_candidate(env, forward, lateral, q2_delta):
    state, _ = env.reset(seed=0, options={"object_count": 1})
    w0 = wiper_xy(state)
    yaw = -math.pi / 2
    direction = np.array([math.cos(yaw), math.sin(yaw)])
    side = np.array([-direction[1], direction[0]])
    target = w0 - forward * direction + lateral * side
    q2_target = value(state, "robot", "pos_arm_joint2") + q2_delta

    # Simultaneously position the base and shoulder with the fingers open.
    for _ in range(12):
        action = np.zeros(11, np.float32)
        action[:2] = np.clip(target - robot_xy(state), -.1, .1)
        action[2] = np.clip(angle_error(yaw, value(state, "robot", "pos_base_rot")), -.1, .1)
        action[4] = np.clip(q2_target - value(state, "robot", "pos_arm_joint2"), -.1, .1)
        state, _, _, _, _ = env.step(action)
    pre = wiper_xy(state).copy()
    for _ in range(4):
        action = np.zeros(11, np.float32)
        action[10] = 1
        state, _, _, _, _ = env.step(action)
    grip = value(state, "robot", "pos_gripper")

    # Retreat opposite the handle. A grasp follows; mere contact stays put.
    for _ in range(4):
        action = np.zeros(11, np.float32)
        action[:2] = -.08 * direction
        action[10] = 1
        state, _, _, _, _ = env.step(action)
    follow = float(np.linalg.norm(wiper_xy(state) - pre))
    disturbed = float(np.linalg.norm(wiper_xy(state) - w0))
    return grip, follow, disturbed, robot_xy(state), wiper_xy(state)


if __name__ == "__main__":
    env = make_env()
    # Current estimate is forward=.40, lateral=-.36. Search local offsets first.
    # Fine XY alignment from renders, but sweep shoulder height broadly.  The
    # vertical handle is only 34 cm tall, so height is the dominant uncertainty.
    candidates = itertools.product((.36, .40, .44),
                                   (-.40, -.36, -.32),
                                   np.arange(-1.4, 2.01, .2))
    for i, candidate in enumerate(candidates):
        result = run_candidate(env, *candidate)
        print(i, "p", candidate, "grip/follow/dist", np.round(result[:3], 4),
              "base", np.round(result[3], 3), "w", np.round(result[4], 3),
              flush=True)
        if result[1] > .03:
            print("LIKELY GRASP", candidate, result, flush=True)
            break
    env.close()
