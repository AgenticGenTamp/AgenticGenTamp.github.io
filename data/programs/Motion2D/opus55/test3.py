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
    t=time.time(); ap.reset(obs, info)
    steps=0; done=False; stuck=0
    while steps < env.max_steps:
        a = ap.get_action(obs)
        obs, r, term, trunc, _ = env.step(a); steps+=1
        stuck += ap.stuck>0
        if term: done=True; break
        if trunc: break
    el=time.time()-t
    res.append((seed, done, steps, stuck, round(el,1), round(ap.margin,4)))
    if not done or stuck or el>5: print(res[-1], flush=True)
print('success', sum(r[1] for r in res), '/', len(res), 'mean steps', np.mean([r[2] for r in res]), 'max t', max(r[4] for r in res))
