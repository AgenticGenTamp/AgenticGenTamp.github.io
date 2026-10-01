import numpy as np
from env_client import make_env
from approach import GeneratedApproach

for count in (0, 1, 2):
    wins = 0
    for seed in range(5):
        env = make_env()
        state, info = env.reset(seed=100 + seed, options={"object_count": count})
        policy = GeneratedApproach(env.action_space, env.observation_space, {})
        policy.reset(state, info)
        for step in range(env.max_steps):
            action = np.asarray(policy.get_action(state), dtype=env.action_space.dtype)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        wins += int(terminated)
        print(count, seed, terminated, step + 1, flush=True)
        env.close()
    print("COUNT", count, wins, "/5", flush=True)
