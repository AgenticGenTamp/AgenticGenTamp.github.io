import sys, math, time
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); seed=int(sys.argv[1])
obs, info = env.reset(seed=seed, options={'object_count':8})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
print('layered', ap.layered, 'margin', ap.margin, [(round(t,2), len(p), p[0], p[-1]) for t,p in ap.segments])
for o in sorted(ap.obstacles): print([round(v,3) for v in o])
print('target', [round(v,3) for v in ap.target])
r=ap.robot
for t in range(150):
    a=ap.get_action(obs)
    obs,rew,term,*_=env.step(a)
    s=[round(float(obs.get(r,f)),4) for f in ['x','y','theta']]
    if term: print('done',t); break
    if ap.stuck or ap.idle or t%10==0: print(t, np.round(a,4), s, 'seg',ap.seg, ap.stuck, ap.idle, round(ap.margin,4))
