"""Log the exact joint trajectory around the visually lowest random-search frame."""
from env_client import make_env
from diag_random.approach import GeneratedApproach

env = make_env(); state, info = env.reset(seed=1)
policy = GeneratedApproach(env.action_space, env.observation_space, {})
policy.reset(state, info)
for step in range(666):
    state, *_ = env.step(policy.get_action(state))
    if 645 <= step <= 665:
        r = state.get_object_from_name("robot")
        q = [float(state.get(r, f"pos_arm_joint{i}")) for i in range(1, 8)]
        print(step, [round(x, 5) for x in q], flush=True)
env.close()
