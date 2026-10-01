import sys, time
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
seeds = range(int(sys.argv[1]), int(sys.argv[2]))
env = make_env()
res=[]
for seed in seeds:
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    plan_steps = len(ap.path)/5 if ap.path else -1
    steps=0; done=False; stuck=0
    while steps < env.max_steps:
        a = ap.get_action(obs)
        obs, r, term, trunc, _ = env.step(a); steps+=1
        stuck += ap.stuck>0
        if term: done=True; break
        if trunc: break
    res.append((seed, done, steps, round(plan_steps,1), stuck))
    if not done or stuck or steps>plan_steps+3: print(res[-1], flush=True)
print('success', sum(r[1] for r in res), '/', len(res), 'mean steps', np.mean([r[2] for r in res]))
