"""Force one guarded box nudge after the current seed-0 policy."""

import numpy as np
from env_client import make_env
from approach import GeneratedApproach


env = make_env(); s, info = env.reset(seed=0)
agent = GeneratedApproach(env.action_space, env.observation_space, {})
agent.reset(s, info)
for k in range(500):
    s, r, term, trunc, info = env.step(agent.get_action(s))
    if term or trunc or agent.phase >= 11:
        break
print("before", k, r, term, np.round(s[[0,1,2,16,17,18,32,33,34],],4).tolist())
for j in range(3):
    a = np.zeros(11, np.float32); a[1] = -.1; a[10] = 0
    s, r, term, trunc, info = env.step(a)
    print("nudge", j, r, term, np.round(s[[0,1,2,16,17,18,32,33,34],],4).tolist())
    if term or trunc:
        break
env.close()
