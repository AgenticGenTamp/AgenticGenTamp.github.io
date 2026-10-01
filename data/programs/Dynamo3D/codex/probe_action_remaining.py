"""Probe all actuator components in one episode to reduce server overhead."""

import numpy as np

from env_client import make_env


FEATURES = (
    "pos_base_x", "pos_base_y", "pos_base_rot",
    "pos_arm_joint1", "pos_arm_joint2", "pos_arm_joint3",
    "pos_arm_joint4", "pos_arm_joint5", "pos_arm_joint6",
    "pos_arm_joint7", "pos_gripper",
)


def values(state):
    robot = state.get_object_from_name("robot")
    return np.array([state.get(robot, f) for f in FEATURES], dtype=float)


def main():
    env = make_env()
    state, info = env.reset(seed=17)
    print("initial", np.round(values(state), 5), info, flush=True)
    previous = values(state)
    for dim in range(11):
        action = np.zeros(11, dtype=np.float32)
        action[dim] = 1.0 if dim == 10 else 0.1
        state, reward, term, trunc, _ = env.step(action)
        now = values(state)
        print("plus", dim, "delta", np.round(now - previous, 5), "r", reward, flush=True)
        previous = now
        action[:] = 0
        state, reward, term, trunc, _ = env.step(action)
        now = values(state)
        print("zero", dim, "delta", np.round(now - previous, 5), "r", reward, flush=True)
        previous = now
    # Gripper toggling: held open, held close.
    for command in (1.0, 1.0, 0.0, 0.0):
        action = np.zeros(11, dtype=np.float32)
        action[10] = command
        state, reward, term, trunc, _ = env.step(action)
        now = values(state)
        print("grip", command, "value", now[10], "delta", now[10] - previous[10], flush=True)
        previous = now
    env.close()


if __name__ == "__main__":
    main()
