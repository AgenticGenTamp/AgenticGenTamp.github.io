"""Small black-box dynamics probe for Tossing3DEnv.

Each experiment uses a fresh environment with the same seed, so reported
changes can be compared coordinate by coordinate.
"""

import numpy as np

from env_client import make_env


ROBOT_FEATURES = (
    "pos_base_x", "pos_base_y", "pos_base_rot",
    "pos_arm_joint1", "pos_arm_joint2", "pos_arm_joint3",
    "pos_arm_joint4", "pos_arm_joint5", "pos_arm_joint6",
    "pos_arm_joint7", "pos_gripper",
    "vel_base_x", "vel_base_y", "vel_base_rot",
    "vel_arm_joint1", "vel_arm_joint2", "vel_arm_joint3",
    "vel_arm_joint4", "vel_arm_joint5", "vel_arm_joint6",
    "vel_arm_joint7", "vel_gripper",
)


def robot_vector(state):
    robot = state.get_object_from_name("robot")
    values = []
    names = []
    for feature in ROBOT_FEATURES:
        try:
            values.append(float(state.get(robot, feature)))
            names.append(feature)
        except (KeyError, ValueError):
            pass
    return names, np.asarray(values)


def object_summary(state):
    rows = []
    for name in state.get_object_names():
        if name == "robot":
            continue
        obj = state.get_object_from_name(name)
        try:
            xyz = tuple(round(float(state.get(obj, f)), 3) for f in ("x", "y", "z"))
        except (KeyError, ValueError):
            continue
        rows.append((name, xyz))
    return rows


def rollout(seed, action, steps):
    env = make_env()
    state, info = env.reset(seed=seed)
    names, initial = robot_vector(state)
    total_reward = 0.0
    terminated = False
    for _ in range(steps):
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    _, final = robot_vector(state)
    env.close()
    return names, initial, final, total_reward, terminated, info


def main():
    seed = 0
    env = make_env()
    initial_state, info = env.reset(seed=seed)
    print("shape", env.action_space.shape, "dtype", env.action_space.dtype,
          "max_steps", env.max_steps)
    print("low ", np.asarray(env.action_space.low))
    print("high", np.asarray(env.action_space.high))
    print("objects", object_summary(initial_state))
    names, robot0 = robot_vector(initial_state)
    print("robot initial", dict(zip(names, np.round(robot0, 4))))
    env.close()

    # Hold each bounded command at its positive limit for ten control steps.
    # Use gripper=initial-like open value for the first ten probes so gripper
    # motion does not obscure their effects.
    for dim in range(11):
        action = np.zeros(18, dtype=np.float32)
        action[10] = 1.0
        action[dim] = 1.0 if dim == 10 else 0.1
        names, before, after, reward, done, _ = rollout(seed, action, 10)
        delta = after - before
        changed = {n: round(float(v), 5) for n, v in zip(names, delta)
                   if abs(v) > 1e-5}
        print(f"dim {dim:2d} x10: changed={changed} reward={reward:.3f} done={done}")

    # Probe the nominally unbounded joint velocity coordinates conservatively.
    for dim in range(11, 18):
        action = np.zeros(18, dtype=np.float32)
        action[10] = 1.0
        action[dim] = 1.0
        names, before, after, reward, done, _ = rollout(seed, action, 10)
        delta = after - before
        changed = {n: round(float(v), 5) for n, v in zip(names, delta)
                   if abs(v) > 1e-5}
        print(f"dim {dim:2d} x10: changed={changed} reward={reward:.3f} done={done}")

    # Compare closing/opening from identical resets over a longer interval.
    for grip in (0.0, 0.5, 1.0):
        action = np.zeros(18, dtype=np.float32)
        action[10] = grip
        names, before, after, reward, done, _ = rollout(seed, action, 30)
        idx = names.index("pos_gripper")
        print(f"gripper={grip:.1f} x30: {before[idx]:.4f} -> {after[idx]:.4f}, "
              f"reward={reward:.3f} done={done}")


if __name__ == "__main__":
    main()
