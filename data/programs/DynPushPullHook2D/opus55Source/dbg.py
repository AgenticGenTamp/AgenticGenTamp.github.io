import sys
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
seed, count = int(sys.argv[1]), int(sys.argv[2]); every = int(sys.argv[3]) if len(sys.argv)>3 else 10
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[count], eval_counts=[count])
s, info = env.reset(seed=seed, options={'object_count': count})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(s, info)
for t in range(300+150*count):
    a = ap.get_action(s)
    if t % every == 0:
        r,h,tb,ob = ap._read(s)
        print(t, ap.phase, getattr(ap,'app_stage',''), getattr(ap,'push_stage',''), len(ap.wps), "R", np.round([r['x'],r['y'],r['th'],r['arm'],r['gap']],2), "H", np.round([h['x'],h['y'],h['th'],h['held']],2), "T", np.round([tb['x'],tb['y']],2), "a", np.round(a,3))
    s, rew, term, trunc, info = env.step(a)
    if term: print("SOLVED", t+1); break
