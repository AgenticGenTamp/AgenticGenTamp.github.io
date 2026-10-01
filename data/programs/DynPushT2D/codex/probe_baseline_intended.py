"""Analyze baseline behavior while correcting only its float32 endpoint overflow."""

import sys

import numpy as np

from approach import GeneratedApproach
from env_client import make_env


def run(seed):
    env = make_env()
    obs, info = env.reset(seed=seed)
    initial = obs.copy()
    policy = GeneratedApproach(env.action_space, env.observation_space, {})
    policy.reset(obs, info)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 200:
        action = np.asarray(policy.get_action(obs), dtype=np.float64)
        action = np.clip(action, env.action_space.low + 1e-7,
                         env.action_space.high - 1e-7)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
    pos_vec = obs[:2] - obs[29:31]
    angle = abs((float(obs[2] - obs[31]) + np.pi) % (2 * np.pi) - np.pi)
    vals = [seed, int(terminated), int(truncated), steps,
            *initial[[0, 1, 2, 16, 17, 29, 30, 31]],
            *obs[[0, 1, 2, 16, 17]], *pos_vec, angle]
    print(" ".join(str(round(float(x), 5)) for x in vals))
    env.close()


for value in sys.argv[1:]:
    run(int(value))
