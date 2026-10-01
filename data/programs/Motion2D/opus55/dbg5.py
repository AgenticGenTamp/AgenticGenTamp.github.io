import sys, math
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); seed=int(sys.argv[1])
obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for o in ap.obstacles: print([round(v,3) for v in o])
r=ap.robot
for t in range(60):
    a=ap.get_action(obs)
    obs,*_=env.step(a)
    print(t, np.round(a,3), [round(float(obs.get(r,f)),3) for f in ['x','y','theta']], ap.stuck)
