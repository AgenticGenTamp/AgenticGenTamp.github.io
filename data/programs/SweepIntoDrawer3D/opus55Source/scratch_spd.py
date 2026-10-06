import numpy as np, sys
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,_=env.reset(seed=0)
for j in (0,1,3,6):
    obs,_=env.reset(seed=0)
    qs=[obs[128+j]]
    for t in range(12):
        a=np.zeros(11); a[3+j]=0.1 if j!=3 else 0.1
        obs,*_=env.step(a); qs.append(obs[128+j])
    print(j, np.round(np.diff(qs),3))
