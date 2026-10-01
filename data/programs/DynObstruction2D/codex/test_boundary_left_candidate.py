import sys

import numpy as np

from boundary_left_candidate import GeneratedApproach
from env_client import make_env


for seed in [int(value) for value in sys.argv[1:]] or [35, 1, 8]:
    env = make_env()
    state, info = env.reset(seed=seed)
    policy = GeneratedApproach(env.action_space, env.observation_space,
                               env.make_primitives())
    policy.reset(state, info)
    for step in range(env.max_steps):
        action = np.asarray(policy.get_action(state), dtype=env.action_space.dtype)
        state, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    block = state.get_objects(env.observation_space.get_type("target_block"))[0]
    print(seed, bool(terminated), step + 1, "special", policy.special,
          "phase", policy.phase, "block",
          round(float(state.get(block, "x")), 3),
          round(float(state.get(block, "y")), 3), flush=True)
    env.close()
