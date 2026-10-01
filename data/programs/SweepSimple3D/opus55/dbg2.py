import numpy as np, sys
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); t0=int(sys.argv[2]); t1=int(sys.argv[3]); every=int(sys.argv[4])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(t1):
    a = ap.get_action(obs)
    if t>=t0 and t%every==0:
        c=ap.cubes.get(ap.target)
        print(t, ap.phase, ap.target, 'th',None if ap.th is None else round(ap.th,2),'base',ap.base.round(3), 'tip',ap._tip_xy().round(3),'cube', None if c is None else c[:3].round(3), 'a',a[:3].round(3), 'dq', np.abs(ap.q-ap.q_hover).max().round(3))
    obs, r, term, trunc, info = env.step(a)
    if term: print('TERM',t); break
