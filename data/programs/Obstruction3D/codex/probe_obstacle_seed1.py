"""Trace seed-1 obstruction transfer and release without changing the policy."""
from env_client import make_env
from approach import GeneratedApproach


def get(state, name, feature):
    return float(state.get(state.get_object_from_name(name), feature))


env = make_env()
state, info = env.reset(seed=1)
policy = GeneratedApproach(env.action_space, env.observation_space, {})
policy.reset(state, info)
for step in range(55):
    before_stage = policy.stage
    before = (get(state, "robot", "joint_2"),
              get(state, "robot", "joint_4"),
              get(state, "obstruction0", "pose_x"),
              get(state, "obstruction0", "pose_y"),
              get(state, "obstruction0", "pose_z"),
              get(state, "robot", "grasp_active"),
              get(state, "robot", "finger_state"))
    action = policy.get_action(state)
    selected = tuple(float(action[i]) for i in (4, 6, 10))
    state, _, terminated, truncated, _ = env.step(action)
    after = (get(state, "robot", "joint_2"),
             get(state, "robot", "joint_4"),
             get(state, "obstruction0", "pose_x"),
             get(state, "obstruction0", "pose_y"),
             get(state, "obstruction0", "pose_z"),
             get(state, "robot", "grasp_active"),
             get(state, "robot", "finger_state"))
    if 15 <= step or before[5] > .5 or after[5] > .5:
        print(step + 1, before_stage, "->", policy.stage,
              "before", tuple(round(x, 5) for x in before),
              "a(q2,q4,g)", tuple(round(x, 5) for x in selected),
              "after", tuple(round(x, 5) for x in after))
    if policy.i > 0 or terminated or truncated:
        break
env.close()
