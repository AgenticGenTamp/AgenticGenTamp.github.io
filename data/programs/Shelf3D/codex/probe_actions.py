"""Small black-box probes for Shelf3D action semantics."""

import numpy as np

from env_client import make_env


ROBOT_FEATURES = [
    "pos_base_x", "pos_base_y", "pos_base_rot",
    "pos_arm_joint1", "pos_arm_joint2", "pos_arm_joint3", "pos_arm_joint4",
    "pos_arm_joint5", "pos_arm_joint6", "pos_arm_joint7", "pos_gripper",
    "vel_base_x", "vel_base_y", "vel_base_rot",
    "vel_arm_joint1", "vel_arm_joint2", "vel_arm_joint3", "vel_arm_joint4",
    "vel_arm_joint5", "vel_arm_joint6", "vel_arm_joint7", "vel_gripper",
]


def robot_vector(state):
    robot = state.get_object_from_name("robot")
    return np.array([state.get(robot, f) for f in ROBOT_FEATURES], dtype=float)


def object_summary(state):
    result = []
    for name in sorted(n for n in state.get_object_names() if n.startswith("cube")):
        obj = state.get_object_from_name(name)
        result.append((obj.name, *(state.get(obj, f) for f in ("x", "y", "z", "bb_x", "bb_y", "bb_z"))))
    return result


def isolated(seed, dim, value, repeats):
    env = make_env()
    state, info = env.reset(seed=seed)
    before = robot_vector(state)
    rewards = []
    a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    a[dim] = value
    for _ in range(repeats):
        state, reward, terminated, truncated, info = env.step(a)
        rewards.append(reward)
        if terminated or truncated:
            break
    after = robot_vector(state)
    env.close()
    return before, after, rewards, info


def main():
    env = make_env()
    state, info = env.reset(seed=0)
    print("space", env.action_space.shape, env.action_space.low.tolist(), env.action_space.high.tolist(), env.max_steps)
    print("names", state.get_object_names())
    print("robot0", np.round(robot_vector(state), 4).tolist())
    print("objects0", object_summary(state))
    print("info0", info)
    env.close()

    for repeats in (1, 10):
        print("repeats", repeats)
        for dim in range(11):
            before, after, rewards, info = isolated(0, dim, 1.0 if dim == 10 else 0.1, repeats)
            delta = after - before
            changed = [(ROBOT_FEATURES[i], round(x, 5)) for i, x in enumerate(delta) if abs(x) > 1e-5]
            print(dim, changed, "reward", round(sum(rewards), 4), "last_info", info)


if __name__ == "__main__":
    main()
