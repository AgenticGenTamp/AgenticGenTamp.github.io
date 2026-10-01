import sys
from env_client import make_env
from approach import GeneratedApproach

counts = [int(sys.argv[1])] if len(sys.argv) > 1 else [0, 1, 2, 3, 4, 5]
seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
for count in counts:
    for seed in range(seeds):
        env = make_env()
        state, info = env.reset(seed=seed, options={"object_count": count})
        agent = GeneratedApproach(env.action_space, env.observation_space, env.make_primitives())
        agent.reset(state, info)
        for step in range(env.max_steps):
            state, _, term, trunc, _ = env.step(agent.get_action(state))
            if term or trunc:
                break
        print(count, seed, term, step + 1)
        env.close()
