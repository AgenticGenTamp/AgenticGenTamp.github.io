import sys, numpy as np
from env_client import make_env
import approach, cfgs
for k, v in getattr(cfgs, sys.argv[1]).items(): setattr(approach, k, v)
env = make_env()
for s in range(100, 110):
    obs, info = env.reset(seed=s, options={'object_count': 1})
    ap = approach.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
    print(s, 'zc', round(ap.shelf_z,3), 'q_place', np.round(ap.q_place,2), 'err', getattr(ap,'place_err',None))
