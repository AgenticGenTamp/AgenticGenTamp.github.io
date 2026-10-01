"""Sweep individual arm joints and report any cube contact/reward transition."""

import numpy as np

from env_client import make_env


def xyz(state):
    obj = state.get_object_from_name("cube1")
    return np.array([state.get(obj, f) for f in ("x", "y", "z")], float)


env = make_env()
for dim in range(3, 10):
    for sign in (-1, 1):
        state, _ = env.reset(seed=123, options={"object_count": 1})
        p0 = xyz(state)
        action = np.zeros(11, np.float32)
        action[dim] = sign * .1
        action[10] = 1.0
        max_delta = 0.0
        unique_rewards = []
        for step in range(50):
            state, reward, term, trunc, _ = env.step(action)
            max_delta = max(max_delta, float(np.linalg.norm(xyz(state) - p0)))
            if not unique_rewards or reward != unique_rewards[-1]:
                unique_rewards.append(reward)
            if term or trunc:
                break
        print(dim, sign, "cube", np.round(xyz(state), 4), "motion", round(max_delta, 4),
              "rewards", unique_rewards, "done", term, trunc)

rng = np.random.default_rng(7)
for trial in range(20):
    state, _ = env.reset(seed=123, options={"object_count": 1})
    p0 = xyz(state)
    max_delta = 0.0
    unique_rewards = []
    for step in range(150):
        if step % 10 == 0:
            action = np.zeros(11, np.float32)
            action[3:10] = rng.choice((-0.1, 0.1), 7)
            action[10] = rng.choice((0.0, 1.0))
        state, reward, term, trunc, _ = env.step(action)
        max_delta = max(max_delta, float(np.linalg.norm(xyz(state) - p0)))
        if not unique_rewards or reward != unique_rewards[-1]:
            unique_rewards.append(reward)
        if term or trunc:
            break
    if max_delta > .001 or len(unique_rewards) > 1:
        print("RANDOM", trial, "cube", np.round(xyz(state), 4),
              "motion", round(max_delta, 4), "rewards", unique_rewards,
              "done", term, trunc)
env.close()
