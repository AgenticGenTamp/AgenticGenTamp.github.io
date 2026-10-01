import numpy as np
from env_client import make_env
import approach
env = make_env()
obs, info = env.reset(seed=707, options={'object_count': 3})
ap = approach.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
prev = None
for n in range(170):
    a = ap.get_action(obs)
    if ap.phase == 'descend' and n > 140:
        r = ap._parse(obs)[0]; q = r[3:10]
        print(n, np.round(q - prev, 4) if prev is not None else None, np.round(approach.qerr(ap.q_t, q), 3), np.round(ap.base_t - r[:3], 3))
    prev = ap._parse(obs)[0][3:10].copy()
    obs, *_ = env.step(a)
