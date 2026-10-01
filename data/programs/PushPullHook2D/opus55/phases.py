import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
from collections import Counter
tot=Counter()
for seed in range(int(sys.argv[1]),int(sys.argv[2])):
    env=make_env(); obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    c=Counter()
    for t in range(1000):
        a=ap.get_action(obs); c[getattr(ap,'phase','?')]+=1
        obs,r,te,tr,_=env.step(a)
        if te: break
    print(seed, t+1, dict(c)); tot+=c
print(tot)
