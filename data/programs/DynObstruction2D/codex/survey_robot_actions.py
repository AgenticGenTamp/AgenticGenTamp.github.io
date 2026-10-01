"""Measure one-step robot state response to each control coordinate."""

import numpy as np

from env_client import make_env


ROBOT_FEATURES = (
    "x", "y", "theta", "arm_joint", "arm_length", "finger_gap",
    "vx_base", "vy_base", "omega_base", "vx_arm", "vy_arm",
    "vx_gripper_l", "vy_gripper_l", "vx_gripper_r", "vy_gripper_r",
)


def robot_values(env, state):
    robot = state.get_objects(env.observation_space.get_type("kin_robot"))[0]
    return np.array([state.get(robot, f) for f in ROBOT_FEATURES])


def main():
    env = make_env()
    base, _ = env.reset(seed=0, options={"object_count": 0})
    before = robot_values(env, base)
    print("features", ROBOT_FEATURES)
    print("initial", np.round(before, 6))
    # Stay just inside bounds: the server's native space uses lower precision.
    for axis, amount in enumerate((0.049, 0.049, 0.19, 0.099, 0.019)):
        state, _ = env.reset(seed=0, options={"object_count": 0})
        action = np.zeros(5)
        action[axis] = amount
        state, reward, terminated, truncated, _ = env.step(action)
        after = robot_values(env, state)
        print("axis", axis, "delta", np.round(after - before, 6),
              "reward", reward, "done", terminated, truncated)
        for _ in range(9):
            state, reward, terminated, truncated, _ = env.step(action)
        after10 = robot_values(env, state)
        print("axis", axis, "delta10", np.round(after10 - before, 6))
    for axis, amount in ((3, -0.099), (4, -0.019)):
        state, _ = env.reset(seed=0, options={"object_count": 0})
        action = np.zeros(5)
        action[axis] = amount
        for _ in range(5):
            state, reward, terminated, truncated, _ = env.step(action)
        print("axis", axis, "negative_delta5", np.round(robot_values(env, state) - before, 6))
    env.close()


if __name__ == "__main__":
    main()
