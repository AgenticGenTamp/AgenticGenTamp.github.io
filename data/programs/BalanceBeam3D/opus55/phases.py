import sys, numpy as np
from collections import defaultdict
from env_client import make_env
from approach import GeneratedApproach
seeds = range(int(sys.argv[1]), int(sys.argv[2]))
tot = defaultdict(float)
for seed in seeds:
    env = make_env(); obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    k = 0
    for t in range(1000):
        a = ap.get_action(obs)
        key = ap.phase + ("0" if k == 0 and ap.phase == "approach" else "")
        tot[key] += 1
        obs, r, te, tr, info = env.step(a)
        if ap.phase == "descend": k = 1
        if te: break
    env.close()
for kk, v in tot.items(): print(kk, v / len(seeds))
