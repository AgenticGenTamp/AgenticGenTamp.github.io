import sys, math, time
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); seed=int(sys.argv[1])
obs, info = env.reset(seed=seed, options={'object_count':8})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
print('plan', ap.path is not None, ap.margin, ap.ptheta, len(ap.path) if ap.path else 0)
for o in sorted(ap.obstacles): print([round(v,3) for v in o])
print('target', [round(v,3) for v in ap.target])
r=ap.robot
prev=None
for t in range(400):
    a=ap.get_action(obs)
    obs,rew,term,*_=env.step(a)
    s=[round(float(obs.get(r,f)),3) for f in ['x','y','theta']]
    if term: print('done',t); break
    if t%20==0 or ap.stuck or ap.idle: print(t, np.round(a,3), s, ap.stuck, ap.idle, ap.path is not None)
