import numpy as np, sys, kin
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); cnt=int(sys.argv[2]); t0=int(sys.argv[3]); t1=int(sys.argv[4]); ev=int(sys.argv[5])
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':cnt})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(t1):
    a = ap.get_action(obs)
    if t>=t0 and t%ev==0:
        p,R=kin.fk(*ap.base,ap.q)
        print(t,ap.phase,'q',ap.q.round(2),'qt',ap.qh_t.round(2),'a',a[3:10].round(3),'tipz',p.round(2))
    obs, r, term, trunc, info = env.step(a)
    if term: break
