import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
for s in range(20):
    obs,info=env.reset(seed=s)
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info); ap.verbose=True
    print("=== seed",s, file=sys.stderr)
    for k in range(1000):
        a=ap.get_action(obs); obs,r,term,tr,info=env.step(a)
        if term: break
env.close()
