import sys
from env_client import make_env
from approach import GeneratedApproach

start, count = map(int, sys.argv[1:3])
failures = []
steps = []
for seed in range(start, start + count):
    env = make_env()
    state, info = env.reset(seed=seed)
    agent = GeneratedApproach(env.action_space, env.observation_space, env.make_primitives())
    agent.reset(state, info)
    for step in range(env.max_steps):
        state, _, term, trunc, _ = env.step(agent.get_action(state))
        if term or trunc:
            break
    if not term:
        failures.append(seed)
    steps.append(step + 1)
    env.close()
print("range", start, start + count, "failures", failures,
      "mean_steps", round(sum(steps) / len(steps), 2), "max_steps", max(steps))
