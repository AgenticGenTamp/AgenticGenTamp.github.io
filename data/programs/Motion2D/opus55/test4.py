import sys, time, math
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env()
for seed in [int(s) for s in sys.argv[1:]]:
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    p=ap.path; c=sum(max(abs(a[0]-b[0]),abs(a[1]-b[1])) for a,b in zip(p,p[1:]))
    steps=0
    while steps<1000:
        obs,r,term,trunc,_=env.step(ap.get_action(obs)); steps+=1
        if term: break
    print(seed, 'plan', round(c/0.05,2), 'steps', steps, 'theta', round(ap.ptheta,2))
