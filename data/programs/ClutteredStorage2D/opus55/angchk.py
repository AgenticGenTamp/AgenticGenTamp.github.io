import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
print(type(obs), getattr(obs,'shape',None))
for t in range(1000):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    ap._parse(obs)
    ths=[('robot',ap.rth)]+[(n,b['th']) for n,b in ap.blocks.items()]
    for n,th in ths:
        if abs(th) > np.pi: print(t, n, repr(th), 'action', a); break
    if term: print('done',t); break
