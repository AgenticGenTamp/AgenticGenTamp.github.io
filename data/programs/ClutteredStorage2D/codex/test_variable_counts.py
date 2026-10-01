"""Exercise explicitly pinned object counts against the live server."""
import sys
from env_client import make_env
from approach import GeneratedApproach

counts = [int(x) for x in sys.argv[1:]] or [1, 3, 5, 7, 9]
for count in counts:
    for seed in range(3):
        env = make_env()
        state, info = env.reset(seed=seed, options={"object_count": count})
        policy = GeneratedApproach(env.action_space, env.observation_space,
                                   env.make_primitives())
        policy.reset(state, info)
        term = trunc = False
        for step in range(1000):
            state, _, term, trunc, _ = env.step(policy.get_action(state))
            if term or trunc:
                break
        print(count, seed, step + 1, "OK" if term else "FAIL", policy.stage)
        env.close()
