import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
from util import objs, rob
seed=int(sys.argv[1]); N=int(sys.argv[2]); M=int(sys.argv[3])
env = make_env()
ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=seed); ap.reset(obs, info)
for t in range(N+M):
    a = ap.get_action(obs)
    obs, r, term, trunc, info = env.step(np.asarray(a, dtype=float))
    if t>=N: print(t, ap.phase, np.round(a,3), rob(obs).round(3), objs(obs))
