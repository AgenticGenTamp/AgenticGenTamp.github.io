import sys, numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")
for s in map(int, sys.argv[1:]):
    o,_=env.reset(seed=s); print(s, "rob",np.round(o[:3],2),"hook",np.round(o[9:12],2),"btn",np.round(o[20:22],2),"tgt",np.round(o[29:31],2))
