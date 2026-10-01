import numpy as np, sys, collections
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); cnt=int(sys.argv[2])
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':cnt})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
C=collections.Counter(); fails=0
for t in range(1000):
    a = ap.get_action(obs); C[ap.phase]+=1
    obs, r, term, trunc, info = env.step(a)
    if term: break
print(t, dict(C), ap.fails)
