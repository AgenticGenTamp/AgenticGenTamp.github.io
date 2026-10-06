import sys
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
seed, count, t0, t1 = [int(a) for a in sys.argv[1:5]]
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[count], eval_counts=[count])
s, info = env.reset(seed=seed, options={'object_count': count})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
for t in range(t1):
    a = ap.get_action(s)
    if t >= t0:
        r,h,tb,ob = ap._read(s)
        C = ap._corner(r) if ap.off is not None else None
        print(t, ap.phase, "R", np.round([r['x'],r['y'],r['th'],r['arm']],3), "H", np.round([h['x'],h['y'],h['th']],3), "Cpred", None if C is None else np.round(C,3), "T", np.round([tb['x'],tb['y'],tb['th']],2), "O", [np.round([o['x'],o['y']],2).tolist() for o in ob], "a", np.round(a,3))
    s, rew, term, trunc, info = env.step(a)
    if term: print("SOLVED", t+1); break
