"""Focused reward/target probe; never imported by the submitted policy."""
from env_client import make_env
from approach import GeneratedApproach
import numpy as np


def xyz(state, name):
    obj = state.get_object_from_name(name)
    return np.array([state.get(obj, f) for f in ("x", "y", "z")])


env = make_env()
state, info = env.reset(seed=0, options={"object_count": 4})
policy = GeneratedApproach(env.action_space, env.observation_space, {})
policy.reset(state, info)
# Repeat the calibrated two-axis cycle on one pair until it overlaps; this is
# more diagnostic than spending steps on the other three pairs.
policy.cube_names = ["cube4"]
bins0 = {c: xyz(state, "bin_" + c) for c in ("red", "green", "blue", "yellow")}
last_reward = None
for step in range(350):
    action = policy.get_action(state)
    # On the final segment of the bin-to-cube pass, drive 5 cm farther in
    # world +y than the baseline's conservative collision-limited endpoint.
    if (step // 166) % 2 == 1 and step % 166 >= 146:
        action[1] = 0.1
    state, reward, term, trunc, _ = env.step(action)
    interesting = reward != last_reward or (step + 1) % 83 == 0 or term
    if interesting:
        rows = []
        for number in range(1, 5):
            color = ("red", "green", "blue", "yellow")[(number - 1) % 4]
            cube = xyz(state, "cube%d" % number)
            now = xyz(state, "bin_" + color)
            rows.append((number, color,
                         round(float(np.linalg.norm(cube - bins0[color])), 3),
                         round(float(np.linalg.norm(cube - now)), 3),
                         tuple(np.round(cube[:2], 3)),
                         tuple(np.round(now[:2], 3))))
        print(step + 1, "reward", reward, "term", term, "dist(init,current)", rows)
    last_reward = reward
    if term or trunc:
        break
env.close()
