import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
sd = int(sys.argv[1]); lo, hi = int(sys.argv[2]), int(sys.argv[3])
env = make_env(); obs, info = env.reset(seed=sd)
ap = GeneratedApproach(env.action_space, env.observation_space, None); ap.reset(obs, info)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for t in range(1, hi + 1):
    a = ap.get_action(obs)
    if t >= lo: print(t, a)
    obs, r, term, trunc, info = env.step(a)
    if term or trunc: break
