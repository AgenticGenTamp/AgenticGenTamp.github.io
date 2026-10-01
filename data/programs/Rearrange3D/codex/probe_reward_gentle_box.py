"""Test whether an upright, gently pushed box activates the target predicate."""
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

env = make_env(); s, info = env.reset(seed=11)
agent = GeneratedApproach(env.action_space, env.observation_space, {}); agent.reset(s, info)
last_y = float(s[17])
for k in range(180):
    a = agent.get_action(s)
    if agent.phase == 10:
        a[:] = 0; a[1] = -.07; a[10] = 1
    s, r, term, trunc, _ = env.step(a)
    if agent.phase == 10 and (abs(float(s[17])-last_y) > .002 or r != -1):
        print(k+1, "r", r, "base_y", round(float(s[94]),3), "box", np.round(s[16:23],4).tolist())
        last_y = float(s[17])
    # Stop at the predicted +Y tangent slot before overrunning it.
    if agent.phase == 10 and s[17] <= s[1] + s[14] + s[30] + .03:
        for _ in range(8):
            s, r, term, trunc, _ = env.step(np.zeros(11, np.float32))
        break
print("final", k+1, "r",r,term,"bowl",np.round(s[:3],3).tolist(),
      "box",np.round(s[16:23],4).tolist(),"can",np.round(s[32:39],4).tolist())
env.close()
