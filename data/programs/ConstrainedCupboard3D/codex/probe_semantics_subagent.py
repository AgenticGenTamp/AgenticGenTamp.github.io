"""Compact one-environment diagnostic used to infer black-box control semantics."""

import numpy as np
from env_client import make_env


def values(state):
    robot = state.get_object_from_name("robot")
    names = ["pos_base_x", "pos_base_y", "pos_base_rot"]
    names += [f"pos_arm_joint{i}" for i in range(1, 8)]
    names += ["pos_gripper"]
    return np.array([state.get(robot, name) for name in names])


env = make_env()
state, info = env.reset(seed=7)
print("meta", info, env.action_space.shape, env.action_space.low, env.action_space.high,
      "max_steps", env.max_steps, flush=True)
for name in sorted(state.get_object_names()):
    obj = state.get_object_from_name(name)
    print(name, obj.type.name, np.round(state.data[obj], 5), flush=True)
print("robot_start", np.round(values(state), 5), flush=True)

# Apply one positive pulse and then one negative pulse on each continuous channel.
for channel in range(11):
    for sign in (1.0, -1.0) if channel < 10 else (1.0, 0.0):
        action = np.zeros(11, dtype=np.float32)
        action[channel] = 0.1 * sign if channel < 10 else sign
        before = values(state)
        state, reward, term, trunc, step_info = env.step(action)
        print("pulse", channel, sign, "delta", np.round(values(state)-before, 5),
              "pos", np.round(values(state), 5), "r", reward, "done", term, trunc,
              "info", step_info, flush=True)
env.close()
