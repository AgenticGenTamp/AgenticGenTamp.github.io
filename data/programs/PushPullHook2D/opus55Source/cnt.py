import numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")
n=0; m=0
for s in range(1000):
    o,_ = env.reset(seed=s)
    if o[21] <= 1.41: n+=1
    if o[21] <= 1.41 and o[30] <= 1.5: m+=1
print(n, m)
