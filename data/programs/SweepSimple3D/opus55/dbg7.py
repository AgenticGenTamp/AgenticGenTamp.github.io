import numpy as np, sys, kin
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); t0=int(sys.argv[2]); t1=int(sys.argv[3]); ev=int(sys.argv[4])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(t1):
    a = ap.get_action(obs)
    if t>=t0 and t%ev==0:
        w=ap.others['wiper_0']
        print(t,ap.phase,'base',ap.base.round(3),'tip',ap._tip_xy().round(3),'wiper',w.round(2),'a',a[:3].round(3))
    obs, r, term, trunc, info = env.step(a)
