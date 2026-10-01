from env_client import make_env
from variant_agent import GeneratedApproach
import numpy as np
import sys

seed = int(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 400
env = make_env()
s, info = env.reset(seed=seed)
p = GeneratedApproach(env.action_space, env.observation_space, {})
p.reset(s, info)
if len(sys.argv) > 3:
    p.horiz_yoff = float(sys.argv[3])
last = None
print("init", seed, "r", np.round(s[:3], 3), "h", np.round(s[9:12], 3),
      "b", np.round(s[20:22], 3), "t", np.round(s[29:31], 3),
      "elbow", p.use_elbow)
best = 99.0
for k in range(limit):
    a = p.get_action(s)
    s, reward, term, trunc, info = env.step(a)
    d = float(np.linalg.norm(s[20:22] - s[29:31]))
    best = min(best, d)
    if p.phase != last or k % 25 == 0 or term or trunc:
        print(k, "ph", p.phase, "age", p.age, "stale", p.stale,
              "r", np.round(s[:2], 3), "h", np.round(s[9:12], 3),
              "b", np.round(s[20:22], 3), "d", round(d, 3),
              "best", round(best, 3), "elbow", p.use_elbow)
        last = p.phase
    if term or trunc:
        break
env.close()
