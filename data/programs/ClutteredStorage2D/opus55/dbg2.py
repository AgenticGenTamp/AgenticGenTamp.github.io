import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed, oc, T = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
env = make_env()
obs, info = env.reset(seed=seed, options={'object_count': oc})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for t in range(T):
    a = ap.get_action(obs)
    if t >= T-3: print(t, ap.task, ap.phase, a, ap.path, getattr(ap,"goal",None), ap.stuck)
    obs, r, term, trunc, info = env.step(a)
ap._parse(obs)
print('robot', ap.rx, ap.ry, ap.rth, ap.arm, ap.vac)
for n,b in ap.blocks.items(): print(n, b['poly'].round(3).tolist())
