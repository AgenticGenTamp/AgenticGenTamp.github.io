import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env=KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
for s in [0,1]:
    obs,_=env.reset(seed=s)
    print(np.round(obs[0:32],3))
