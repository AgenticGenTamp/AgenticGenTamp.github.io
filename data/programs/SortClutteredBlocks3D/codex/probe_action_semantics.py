"""Small black-box probe for SortClutteredBlocks3D action semantics."""

import numpy as np

from env_client import make_env


def robot_values(state):
    robot = state.get_object_from_name("robot")
    names = (
        "pos_base_x", "pos_base_y", "pos_base_rot",
        "pos_arm_joint1", "pos_arm_joint2", "pos_arm_joint3",
        "pos_arm_joint4", "pos_arm_joint5", "pos_arm_joint6",
        "pos_arm_joint7", "pos_gripper",
    )
    return np.array([state.get(robot, name) for name in names], dtype=float)


def run_case(env, label, action, steps=5):
    state, info = env.reset(seed=0)
    before = robot_values(state)
    print(label, "initial", np.round(before, 4))
    for step in range(steps):
        state, reward, terminated, truncated, info = env.step(action)
        now = robot_values(state)
        print(
            " step", step + 1, "delta", np.round(now - before, 4),
            "now", np.round(now, 4), "r", round(reward, 4),
            "done", terminated or truncated,
        )
    print()


def run_impulse_case(env, label, dim):
    state, info = env.reset(seed=0)
    before = robot_values(state)
    actions = []
    impulse = np.zeros(11, dtype=np.float32)
    impulse[10] = 1.0
    impulse[dim] = 0.1
    actions.append(impulse)
    for _ in range(5):
        hold = np.zeros(11, dtype=np.float32)
        hold[10] = 1.0
        actions.append(hold)
    print(label, "initial", np.round(before, 4))
    for step, action in enumerate(actions, 1):
        state, reward, terminated, truncated, info = env.step(action)
        print(" step", step, "cmd", action[dim], "delta", np.round(robot_values(state) - before, 4))
    print()


def main():
    env = make_env()
    print("shape", env.action_space.shape)
    print("low", env.action_space.low)
    print("high", env.action_space.high)
    zero_open = np.zeros(11, dtype=np.float32)
    zero_open[10] = 1.0
    run_case(env, "zero/open", zero_open, 3)

    for dim, label in ((0, "base+x"), (1, "base+y"), (2, "base+yaw")):
        action = zero_open.copy()
        action[dim] = 0.1
        run_case(env, label, action, 4)

    action = zero_open.copy()
    action[3] = 0.1
    run_case(env, "joint1 +", action, 5)

    action = zero_open.copy()
    action[3:10] = 0.1
    run_case(env, "all joints +", action, 3)

    action = np.zeros(11, dtype=np.float32)
    run_case(env, "zero/closed", action, 5)
    run_impulse_case(env, "base x impulse", 0)
    run_impulse_case(env, "joint1 impulse", 3)
    env.close()


if __name__ == "__main__":
    main()
