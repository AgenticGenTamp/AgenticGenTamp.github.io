import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0")
k = env._kinder_env; oc = k._object_centric_env
obs,_ = env.reset(seed=0)
print(k.metadata.get('observation_space_description',''))
o2, r, term, trunc, _ = env.step(np.zeros(11, dtype=np.float32))
print("reward", r, term, trunc)
