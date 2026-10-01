"""Trace can xy during the current staged pushes to locate target crossings."""

import numpy as np
from env_client import make_env
from approach import GeneratedApproach


env = make_env(); s, info = env.reset(seed=0)
agent = GeneratedApproach(env.action_space, env.observation_space, {})
agent.reset(s, info)
old = s.copy()
for k in range(300):
    prior_phase = agent.phase
    s, r, term, trunc, info = env.step(agent.get_action(s))
    moved = np.linalg.norm(s[32:34] - old[32:34]) > .002
    if moved or r != -1 or agent.phase != prior_phase:
        print(k, "phase", agent.phase, "r", r, "canxy", np.round(s[32:34],4).tolist(),
              "basey", round(float(s[94]),4))
    old = s.copy()
    if agent.phase >= 7 or term or trunc:
        break
# Keep the deployed posture and take three extra lateral actions to test the
# narrow 5 cm x/y target window beyond the policy's conservative stop.
for j in range(6):
    a = np.zeros(11, np.float32); a[1] = .025
    s, r, term, trunc, info = env.step(a)
    print("extra", j, "r", r, "done", term,
          "can", np.round(s[32:39], 4).tolist(),
          "bowlxy", np.round(s[0:2], 4).tolist())
for j in range(20):
    a = np.zeros(11, np.float32)
    s, r, term, trunc, info = env.step(a)
    if j in (0, 4, 9, 19) or r != -1 or term:
        print("settle", j, r, term, np.round(s[32:39], 4).tolist())
    if term or trunc:
        break
for j in range(10):
    a = np.zeros(11, np.float32); a[1] = .01
    s, r, term, trunc, info = env.step(a)
    print("gentle", j, r, term, np.round(s[[0,1,32,33,34,37]], 4).tolist())
    if term or trunc:
        break
env.close()
