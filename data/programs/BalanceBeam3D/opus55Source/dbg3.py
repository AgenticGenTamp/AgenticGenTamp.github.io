import numpy as np, approach
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ap = approach.GeneratedApproach(env.action_space, env.observation_space, {})
obs,info = env.reset(seed=0); ap.reset(obs,info)
for t in range(75):
    a=ap.get_action(obs)
    obs,*_ = env.step(a)
    if t%4==0: print(t, np.round(obs[19:26],2), "act", np.round(a[3:10],2))
