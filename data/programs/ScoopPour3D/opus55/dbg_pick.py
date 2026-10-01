import numpy as np, kin
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=0)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
prev=None
for t in range(300):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    if ap.phase!=prev:
        base,q=ap._robot(obs); p,_=kin.fk_world(base,q)
        c=ap._objs(obs)[0].get(ap.cur) if ap.cur else None
        print(t, ap.phase, ap.pt, ap.cur, 'tool',p.round(3), 'cube', None if c is None else c[:3].round(3), 'qerr',np.abs(q-ap.q_int).max().round(3), 'base',base.round(3))
        prev=ap.phase
