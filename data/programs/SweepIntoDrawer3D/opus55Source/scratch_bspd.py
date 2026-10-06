import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
for s in (0.05,0.08,0.1):
    obs,_=env.reset(seed=0)
    xs=[obs[126]]
    for t in range(10):
        a=np.zeros(11); a[1]=s
        obs,*_=env.step(a); xs.append(obs[126])
    print(s, np.round(np.diff(xs),3))
