"""Probe sparse reward when the can starts at its likely lower bowl slot."""
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

env = make_env()
s, info = env.reset(seed=11)
agent = GeneratedApproach(env.action_space, env.observation_space, {})
agent.reset(s, info)
last_r = None
print("start", np.round(s[[0, 1, 2, 16, 17, 18, 32, 33, 34]], 4).tolist())
for k in range(260):
    old = s.copy()
    s, r, term, trunc, _ = env.step(agent.get_action(s))
    if r != last_r or term or trunc:
        print("event", k + 1, "phase", agent.phase, "r", r, term, trunc,
              "xyz", np.round(s[[0,1,2,16,17,18,32,33,34]], 4).tolist(),
              "quat", np.round(s[[19,20,21,22,35,36,37,38]], 3).tolist())
        last_r = r
    if term or trunc or agent.phase >= 11:
        break
print("final", k + 1, agent.phase, r, term, np.round(s[[0,1,2,16,17,18,32,33,34]],4).tolist(),
      np.round(s[[19,20,21,22,35,36,37,38]],3).tolist())
env.close()
