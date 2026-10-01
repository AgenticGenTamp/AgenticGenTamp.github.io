from env_client import make_env
from approach import GeneratedApproach
import numpy as np
env = make_env(); obs, info = env.reset(seed=159)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap._parse(obs)
r=ap.robot; print('robot', [round(float(obs.get(r,f)),3) for f in ['x','y','theta']])
for o in ap.obstacles: print([round(v,3) for v in o])
print('target', ap.target)
print('start free', ap._free(obs.get(r,'x'), obs.get(r,'y')))
