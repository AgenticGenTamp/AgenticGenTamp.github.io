import sys, numpy as np
from env_client import make_env
from approach import *
env = make_env(); obs,info=env.reset(seed=int(sys.argv[1]))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
B=obs[20:22]; T=obs[29:31]
for op in ap.grasp_options():
    print(round(op['a'],2), op['side'], op['R'].round(3), 'off',op['off'].round(3), round(op['dth'],3), 'npush', len(ap.push_candidates(op['off'],op['dth'],B,T)))
