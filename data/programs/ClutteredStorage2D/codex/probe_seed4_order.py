"""Probe picking a regrasp-corridor blocker before the blocked payload."""

from approach import GeneratedApproach
from env_client import make_env

env = make_env()
state, info = env.reset(seed=4)
policy = GeneratedApproach(env.action_space, env.observation_space,
                           env.make_primitives())
policy.reset(state, info)

for step in range(1600):
    if policy.stage == "choose":
        outside = [b for b in policy._blocks(state) if not policy._inside(state, b)]
        stored = len(policy._blocks(state)) - len(outside)
        if stored >= 4 and outside:
            policy.priority_name = min(outside,
                                       key=lambda b: float(state.get(b, "x"))).name
    if policy.stage == "orient" and policy.target_name:
        stored = sum(policy._inside(state, b) for b in policy._blocks(state))
        if stored >= 4 and abs(policy.grasp_offset) < .20:
            policy.regrasped = True
    action = policy.get_action(state)
    state, _, terminated, truncated, step_info = env.step(action)
    if step % 100 == 99:
        print(step+1, policy.stage, policy.target_name)
    if terminated or truncated:
        print("RESULT", step+1, terminated, truncated, step_info)
        break
else:
    print("RESULT", 1600, False, False, step_info)
print("blocks", [(b.name, round(float(state.get(b, "x")), 3),
                  round(float(state.get(b, "y")), 3))
                 for b in policy._blocks(state)])
env.close()
