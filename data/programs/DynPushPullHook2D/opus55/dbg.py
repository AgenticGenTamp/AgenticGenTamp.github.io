import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); T0=int(sys.argv[2]); T1=int(sys.argv[3]); every=int(sys.argv[4]) if len(sys.argv)>4 else 1
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(T1):
    a=ap.get_action(obs)
    if t>=T0 and (t-T0)%every==0:
        d=ap._parse(obs)
        print(t, ap.phase, 'r',d['r'].round(3), round(d['rth'],3), round(d['arm'],3), round(d['gap'],3),'h',d['h'].round(3), round(d['hth'],3), int(d['held']), 't',d['t'].round(3),'a',np.round(a,3))
    obs,r,term,trunc,info=env.step(a)
    if term: print('TERM',t); break
