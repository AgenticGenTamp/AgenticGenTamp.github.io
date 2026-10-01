import numpy as np, sys
import approach
from env_client import make_env
g=float(sys.argv[1]); approach.PRE_GRIP=g
res=[]
for seed in range(6):
    env=make_env(); obs,info=env.reset(seed=seed, options={'object_count':1})
    ap=approach.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    ok=None
    for t in range(400):
        a=ap.get_action(obs)
        if ap.phase=='turn': ok=True; break
        if ap.phase=='reopen': ok=False; break
        obs,r,term,tr,info=env.step(a)
    res.append(ok)
print(g,res)
