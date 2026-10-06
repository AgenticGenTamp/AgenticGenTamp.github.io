import sys
import numpy as np
from robocode.environments.variable_object_count_env import VariableObjectCountEnv
from approach import GeneratedApproach
count = int(sys.argv[1])
for seed in range(int(sys.argv[2]), int(sys.argv[3])):
    env = VariableObjectCountEnv(constant_object_env_path="kinder.envs.dynamic2d.dyn_pushpullhook2d:DynPushPullHook2DEnv",
        count_kwarg="num_obstructions", count_object_prefix="obstruction", design_counts=[count], eval_counts=[count])
    s, info = env.reset(seed=seed, options={'object_count': count})
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(s, info)
    log = []; last = None; n = 0
    for t in range(300+150*count):
        a = ap.get_action(s)
        key = ap.phase + ("/" + str(ap.app_stage) if ap.phase=="approach" else "") + ("/" + ap.moves[0][0] if ap.phase=="moves" and ap.moves else "") + ("/%d"%len(ap.wps) if ap.phase=="exec" else "")
        if key != last:
            if last: log.append(f"{last}:{n}")
            last = key; n = 0
        n += 1
        s, rew, term, trunc, info = env.step(a)
        if term: break
    log.append(f"{last}:{n}")
    print(seed, t+1, " ".join(log))
