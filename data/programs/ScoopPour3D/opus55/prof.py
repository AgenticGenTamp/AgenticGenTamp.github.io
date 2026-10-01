import time, numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=0)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
ta=0; te=0
for t in range(200):
    t0=time.time(); a=ap.get_action(obs); t1=time.time()
    obs,r,term,trunc,info=env.step(a); t2=time.time(); ta+=t1-t0; te+=t2-t1
print('agent',ta,'env',te)
