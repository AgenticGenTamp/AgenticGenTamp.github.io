"""Small black-box probes for Dynamo3D action semantics."""

import numpy as np

from env_client import make_env


def robot_values(state):
    robot = state.get_object_from_name("robot")
    features = (
        "pos_base_x", "pos_base_y", "pos_base_rot",
        "pos_arm_joint1", "pos_arm_joint2", "pos_arm_joint3",
        "pos_arm_joint4", "pos_arm_joint5", "pos_arm_joint6",
        "pos_arm_joint7", "pos_gripper",
    )
    return np.array([state.get(robot, f) for f in features], dtype=float)


def objects(state):
    out = {}
    for name in state.get_object_names():
        if name.startswith("obstacle_chair"):
            obj = state.get_object_from_name(name)
            out[name] = np.array([state.get(obj, f) for f in ("x", "y", "z")])
    return out


def main():
    seed = 0
    env = make_env()
    state, info = env.reset(seed=seed)
    print("space", env.action_space.shape, env.action_space.low, env.action_space.high,
          "max", env.max_steps)
    print("names", state.get_object_names(), "info", info)
    print("robot0", np.round(robot_values(state), 5))
    print("objects0", {k: np.round(v, 4) for k, v in objects(state).items()})
    env.close()

    # Each dimension gets an independent same-seed episode and ten held steps.
    for dim in range(11):
        env = make_env()
        state, _ = env.reset(seed=seed)
        initial = robot_values(state)
        action = np.zeros(11, dtype=np.float32)
        action[dim] = 1.0 if dim == 10 else 0.1
        rows = []
        total = 0.0
        for t in range(10):
            state, reward, term, trunc, info = env.step(action)
            total += reward
            if t in (0, 1, 4, 9):
                rows.append((t + 1, np.round(robot_values(state) - initial, 4), reward,
                             term, trunc))
        print("dim", dim, "held", rows, "sumreward", round(total, 4),
              "objdelta", {k: np.round(v - objects(state if False else state).get(k, v), 3)
                           for k, v in objects(state).items()})
        env.close()

    # One-step + then - tests to distinguish velocity commands from position deltas.
    for dim in (0, 1, 2, 3, 9, 10):
        env = make_env()
        state, _ = env.reset(seed=seed)
        initial = robot_values(state)
        vals = []
        for sign in (1, 0, -1, 0):
            action = np.zeros(11, dtype=np.float32)
            if dim == 10:
                action[dim] = 1.0 if sign > 0 else 0.0
            else:
                action[dim] = sign * 0.1
            state, reward, term, trunc, _ = env.step(action)
            vals.append(np.round(robot_values(state) - initial, 5))
        print("pulse dim", dim, vals)
        env.close()


if __name__ == "__main__":
    main()
