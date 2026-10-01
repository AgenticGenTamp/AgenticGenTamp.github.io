"""Test a joint-1 lateral arc after the validated yellow x push."""

import numpy as np

from approach import GeneratedApproach
from env_client import make_env


env = make_env()
state, info = env.reset(seed=0, options={"object_count": 4})
policy = GeneratedApproach(env.action_space, env.observation_space, {})
policy.reset(state, info)

# Complete exactly the first (yellow x-axis) policy cycle.
for _ in range(166):
    state, _, _, _, _ = env.step(policy.get_action(state))

robot = state.get_object_from_name("robot")
base_goal = np.array([state.get(robot, f) for f in
                      ("pos_base_x", "pos_base_y", "pos_base_rot")])
# Maintain light radial pressure as the arc's x projection shortens.
base_goal[0] += 0.035
q_start = np.array([state.get(robot, "pos_arm_joint%d" % i)
                    for i in range(1, 8)])
q_goal = np.array([0.0, 1.30, np.pi, -1.70, 0.0, 1.0, 0.0])


def pos(name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, f) for f in ("x", "y", "z")])


print("start", "base", np.round(base_goal, 3), "q", np.round(q_start, 3),
      "cube4", np.round(pos("cube4"), 3),
      "yellow", np.round(pos("bin_yellow"), 3))

# Probe the opposite shoulder direction too; observed kinematic sign is not
# exposed by the object-centric state.
q_goal[0] = 0.24
for step in range(100):
    robot = state.get_object_from_name("robot")
    base = np.array([state.get(robot, f) for f in
                     ("pos_base_x", "pos_base_y", "pos_base_rot")])
    q = np.array([state.get(robot, "pos_arm_joint%d" % i)
                  for i in range(1, 8)])
    action = np.zeros(11, dtype=np.float32)
    action[:3] = np.clip(1.5 * (base_goal - base), -0.1, 0.1)
    action[3:10] = np.clip(1.5 * (q_goal - q), -0.1, 0.1)
    state, reward, _, _, _ = env.step(action)
    if step % 10 == 9:
        print(step + 1, "q1", round(float(q[0]), 3), "reward", reward,
              "cube4", np.round(pos("cube4"), 3),
              "yellow", np.round(pos("bin_yellow"), 3))

# Reverse the arc: from the outside of the now-contacted yellow bin, this
# should sweep the bin inward (+world-y) over the x-aligned cube.
q_goal[0] = 0.0
for step in range(100):
    robot = state.get_object_from_name("robot")
    base = np.array([state.get(robot, f) for f in
                     ("pos_base_x", "pos_base_y", "pos_base_rot")])
    q = np.array([state.get(robot, "pos_arm_joint%d" % i)
                  for i in range(1, 8)])
    action = np.zeros(11, dtype=np.float32)
    action[:3] = np.clip(1.5 * (base_goal - base), -0.1, 0.1)
    action[3:10] = np.clip(1.5 * (q_goal - q), -0.1, 0.1)
    state, reward, _, _, _ = env.step(action)
    if step % 10 == 9:
        print("reverse", step + 1, "q1", round(float(q[0]), 3),
              "reward", reward, "cube4", np.round(pos("cube4"), 3),
              "yellow", np.round(pos("bin_yellow"), 3))

env.close()
