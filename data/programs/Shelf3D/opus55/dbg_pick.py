import sys, numpy as np
from env_client import make_env
import approach
env = make_env()
obs, info = env.reset(seed=int(sys.argv[1]), options={'object_count': int(sys.argv[2])})
ap = approach.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
prev = None
for n in range(int(sys.argv[3])):
    a = ap.get_action(obs)
    if ap.phase != prev:
        r, c, _ = ap._parse(obs)
        print(n, ap.phase, ap.cur, 'k', ap.cur_k, 'base_t', np.round(ap.base_t, 2), 'cube', np.round(c[ap.cur][:3], 3) if ap.cur in c else None)
        prev = ap.phase
    obs, *_ = env.step(a)
