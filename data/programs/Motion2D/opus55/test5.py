import sys, time
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
oc=int(sys.argv[1]); seeds=range(int(sys.argv[2]), int(sys.argv[3]))
env = make_env()
res=[]
for seed in seeds:
    try:
        obs, info = env.reset(seed=seed, options={'object_count':oc})
    except Exception as e:
        print('reset fail', e); break
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    t=time.time(); ap.reset(obs, info); tp=time.time()-t
    steps=0; done=False
    while steps < env.max_steps:
        obs, r, term, trunc, _ = env.step(ap.get_action(obs)); steps+=1
        if term: done=True; break
        if trunc: break
    el=time.time()-t
    res.append((seed, done, steps, round(tp,1), round(el,1), len(ap.obstacles)))
    if not done or el>10: print(res[-1], flush=True)
print('oc',oc,'success', sum(r[1] for r in res), '/', len(res), 'mean steps', np.mean([r[2] for r in res]), 'max t', max(r[4] for r in res), 'nobs', res[-1][5])
