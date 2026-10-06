import numpy as np
from robocode.environments.kinder_geom3d_env import KinderGeom3DEnv
env = KinderGeom3DEnv("kinder/SweepIntoDrawer3D-o5-v0"); k = env._kinder_env; oc = k._object_centric_env
def setst(v): oc._set_state(k.observation_space.devectorize(v))
obs,_ = env.reset(seed=0)
for d, xs in [(0.0, 0.5), (0.4, 1.0), (0.4, 0.6)]:
    s = obs.copy(); s[107]=d
    for i in range(5):
        b=16*i; s[b:b+3]=[xs+0.04*(i-2), 0.0, 0.24]; s[b+3:b+7]=[1,0,0,0]; s[b+7:b+13]=0
    setst(s)
    print("restored drawer", k.observation_space.vectorize(oc._current_state)[107])
    for t in range(5): o,r,term,_,_ = env.step(np.zeros(11,np.float32))
    print(f"drawer={d} cubes x~{xs}: r",r,"term",term, "drawer now", round(o[107],3), "cube xs", np.round(o[[0,16,32,48,64]],3), "z", np.round(o[2],3))
