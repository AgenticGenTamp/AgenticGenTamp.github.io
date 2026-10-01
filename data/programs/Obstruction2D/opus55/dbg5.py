import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed, count, T, N = map(int, sys.argv[1:5])
env = make_env()
obs, info = env.reset(seed=seed, options={'object_count': count} if count else None)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for t in range(T + N):
    a = ap.get_action(obs)
    if t >= T:
        r = ap.robot
        print(t, 'rob', round(r['x'],4), round(r['y'],4), round(r['arm'],4), r['vac'], 'act', np.round(a,4), 'wp', ap.plan[0] if ap.plan else None)
    obs, rew, term, trunc, info = env.step(a)
    if term: print('term', t); break
