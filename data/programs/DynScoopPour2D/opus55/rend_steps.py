import sys, os
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
from myrender import render
seed=int(sys.argv[1]); steps=[int(s) for s in sys.argv[2].split(',')]
os.makedirs('myr',exist_ok=True)
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(max(steps)+1):
    if t in steps: render(obs, f'myr/s{seed}_{t:04d}.png'); print(t, ap.phase)
    a=ap.get_action(obs); obs,rew,term,trunc,_=env.step(a)
    if term: break
