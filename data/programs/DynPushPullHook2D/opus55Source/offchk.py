import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[1], eval_counts=[1])
for seed in range(12):
    s, info = env.reset(seed=seed, options={'object_count': 1})
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
    for t in range(200):
        a = ap.get_action(s)
        if ap.phase == "grasp" and ap.close_count == 1:
            r,h,tb,ob = ap._read(s); u=np.array([np.cos(r['th']),np.sin(r['th'])]); G=np.array([r['x'],r['y']])+r['arm']*u
            E = ap._bar_end(h); pre_off = (E-G)@u
        s, *_ = env.step(a)
        if ap.off is not None:
            print(seed, "end-in-gripper before close %.3f"%pre_off, "off", np.round(ap.off,3), "dth %.3f"%ap.dth); break
