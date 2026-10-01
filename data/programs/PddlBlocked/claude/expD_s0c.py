import numpy as np, math
from expD_lib import *
env = make_env()
for seed in [0,3]:
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    print("seed",seed,"tasks",ap.tasks if hasattr(ap,'tasks') else None)
    for t in range(12):
        a = ap.get_action(obs); prev = np.concatenate([ap._robot(obs)["base"], ap._robot(obs)["q"]])
        obs,_,_,_,_ = env.step(a)
        now = np.concatenate([ap._robot(obs)["base"], ap._robot(obs)["q"]])
        rej = np.max(np.abs(now-prev))<1e-9
        print(f"  t={t} task_i={ap.task_i} qlen={len(ap.queue)} rej={rej} a={np.round(a,3)}")
