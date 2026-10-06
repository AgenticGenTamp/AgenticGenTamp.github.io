import sys, numpy as np
from collections import Counter
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")
tot = Counter(); N=0; replans=0
for s in range(int(sys.argv[1]), int(sys.argv[2])):
    obs, info = env.reset(seed=s)
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    c = Counter()
    for n in range(500):
        a = ap.get_action(obs); c[ap.mode]+=1
        obs, r, term, trunc, info = env.step(a)
        if term: break
    tot.update(c); N+=1
    print(s, dict(c))
print({k: v/N for k,v in tot.items()})
