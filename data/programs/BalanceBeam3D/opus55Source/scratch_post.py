import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
from approach import GeneratedApproach, HOME
env = KinderGeom3DEnv("kinder/BalanceBeam3D-o3-v0")
ap = GeneratedApproach(env.action_space, env.observation_space, {})
np.set_printoptions(precision=3, suppress=True)
print("HOME", np.array(HOME))
for z in (0.13,0.12,0.07,0.03,0.015):
    q=ap.posture(z); print(z, q, "d", q-np.array(HOME), "max", np.abs(q-np.array(HOME)).max())
