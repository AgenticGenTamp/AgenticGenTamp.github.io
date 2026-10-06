import numpy as np, sys
sys.path.insert(0,'.')
from approach import GeneratedApproach
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
obs,info=env.reset(seed=0)
ap=GeneratedApproach(env.action_space, env.observation_space, None)
ap.reset(obs,info)
for t in range(120):
    a=ap.get_action(obs); obs,*_=env.step(a)
    if t%10==9:
        print(t, "q", np.round(obs[128:135],2), "b", np.round(obs[125:128],2), "ee", np.round(ap.ee()[0],3), "dr", round(float(obs[107]),3))
print(np.round(ap.HOOK_Q,2))
