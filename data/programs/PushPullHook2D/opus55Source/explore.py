import time, numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")
for s in range(5):
    obs,_ = env.reset(seed=s)
    print(s, np.round(obs[[0,1,2,9,10,11,20,21,29,30]],3))
t=time.time()
for i in range(200): env.step(env.action_space.sample())
print("step time", (time.time()-t)/200)
print(type(env), [a for a in dir(env) if not a.startswith('__')][:50])
