import time, sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env()
seed=int(sys.argv[1])
oc=int(sys.argv[2]) if len(sys.argv)>2 else None
obs, info = env.reset(seed=seed, options={'object_count':oc} if oc else None)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
ta=0; te=0
for t in range(1000):
    t0=time.time(); a = ap.get_action(obs); t1=time.time()
    obs, r, term, trunc, info = env.step(a); t2=time.time()
    ta+=t1-t0; te+=t2-t1
    if term: break
print('steps',t+1,'agent',ta,'env',te)
