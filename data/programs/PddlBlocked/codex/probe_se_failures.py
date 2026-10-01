"""Focused black-box probes for southeast/no-spare failures."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np

SEEDS = [74, 79, 120, 176, 196]

for seed in SEEDS:
    env = make_env()
    state, info = env.reset(seed=seed)
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(state, info)
    print("seed", seed, "green", policy.green, "block", policy.blocker,
          "out", policy.out, "theta", policy.theta, "off", policy.off,
          "block target", policy.target(policy.blocker), "green target", policy.target(policy.green))
    last = None
    for step in range(80):
        action = policy.get_action(state)
        old = policy.robot(state)
        state, reward, done, truncated, info = env.step(action)
        new = policy.robot(state)
        if policy.stage != last or (np.max(np.abs(new-old)) < 1e-6 and np.max(np.abs(action[:10])) > .01):
            last = policy.stage
            print(" ", step, "stage", policy.stage, "base", new, "held",
                  policy.g(state, "robot", "grasp_active"), "act", np.round(action[:10], 3),
                  "REJECT" if np.max(np.abs(new-old)) < 1e-6 and np.max(np.abs(action[:10])) > .01 else "")
        if done or truncated:
            break
    env.close()
