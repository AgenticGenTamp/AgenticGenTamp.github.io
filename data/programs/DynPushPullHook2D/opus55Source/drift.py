import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach, rot
env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
    count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[1], eval_counts=[1])
for rate in [0.065, 0.03, 0.015]:
  for armrate in [0.0]:
    s, info = env.reset(seed=0, options={'object_count': 1})
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
    while ap.off is None:
        a = ap.get_action(s); s, *_ = env.step(a)
    r,h,tb,ob = ap._read(s)
    # move down a bit
    tot = 0
    for k in range(int(3.0/rate)):
        a = np.array([0,0,-rate*0.999,0,-0.019]); s, *_ = env.step(a); tot += rate
    r,h,tb,ob = ap._read(s)
    u=np.array([np.cos(r['th']),np.sin(r['th'])]); G=np.array([r['x'],r['y']])+r['arm']*u
    print(rate, "rot", round(tot,2), "off now", np.round(rot(-r['th'], np.array([h['x'],h['y']])-G),3), "held", h['held'])
