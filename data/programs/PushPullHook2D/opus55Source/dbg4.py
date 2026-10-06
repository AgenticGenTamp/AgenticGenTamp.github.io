import sys, numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach, compose
env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")
s=int(sys.argv[1]); a0=int(sys.argv[2]); a1=int(sys.argv[3])
obs, info = env.reset(seed=s)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for n in range(a1):
    a = ap.get_action(obs)
    if n>=a0:
        H = compose(obs[[0,1,2]], ap.T) if ap.T is not None else None
        print(n, ap.mode, "rob", np.round(obs[[0,1,2,4,6]],3), "hook", np.round(obs[9:12],3), "pred", None if H is None else np.round(H,3), "btn", np.round(obs[20:22],3), "act", np.round(a,3))
    obs, r, term, trunc, info = env.step(a)
