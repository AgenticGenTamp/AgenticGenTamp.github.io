import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
env._kinder_env._allow_state_access = True
obs,_ = env.reset(seed=0)
s = obs.copy(); s[107] = 0.4
env._kinder_env.set_state(s); print("drawer after set_state", env._kinder_env.get_state()[107])
s2 = env._kinder_env.get_state().copy()
for i in range(5):
    b = 16*i; s2[b:b+3] = [1.0 + 0.05*(i%3) - 0.05, -0.1 + 0.05*i, 0.24]; s2[b+3:b+7]=[1,0,0,0]; s2[b+7:b+13]=0
env._kinder_env.set_state(s2)
for t in range(3):
    o, r, term, tr, _ = env.step(np.zeros(11, np.float32))
    print(t, "r", r, "term", term, "drawer", round(o[107],4), "cubes", np.round([o[16*i:16*i+3] for i in range(5)],3).tolist())
