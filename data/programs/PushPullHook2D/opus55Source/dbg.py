import sys, numpy as np
from robocode.environments.kinder_geom2d_env import KinderGeom2DEnv
from approach import GeneratedApproach
env = KinderGeom2DEnv("kinder/PushPullHook2D-v0")
s=int(sys.argv[1]); MAX=int(sys.argv[2]) if len(sys.argv)>2 else 200
obs, info = env.reset(seed=s)
print("init", np.round(obs[[0,1,2,4,9,10,11,20,21,29,30]],3))
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
pm=None
for n in range(MAX):
    a = ap.get_action(obs)
    if ap.mode!=pm: print(n, "mode", ap.mode, "push", None if ap.push is None else (ap.push[0], np.round(ap.push[1],3), ap.goals), np.round(obs[[0,1,2,20,21]],3)); pm=ap.mode
    prev=obs
    obs, r, term, trunc, info = env.step(a)
    if np.allclose(prev[:3], obs[:3]) and np.abs(a[:3]).max()>1e-4: print(n, "BLOCKED", ap.mode, np.round(a,3), np.round(obs[[0,1,2]],3), ap.plan[:2])
    if term: print("DONE", n); break
