import numpy as np, sys, kin
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); cnt=int(sys.argv[2]); t0=int(sys.argv[3]); t1=int(sys.argv[4]); ev=int(sys.argv[5])
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':cnt})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(t1):
    a = ap.get_action(obs)
    if t>=t0 and t%ev==0:
        c=ap.cubes.get(ap.target)
        print(t,ap.phase,ap.target,'W',ap.others['wiper_0'][:3].round(2),'th',None if ap.th is None else round(ap.th,2),'base',ap.base.round(3),'tip',ap._tip_xy().round(3),'cube',None if c is None else c[:3].round(3),'a',a[:3].round(3),'dq',np.abs(ap.q-ap.qh_t).max().round(3))
    obs, r, term, trunc, info = env.step(a)
    if term: break
