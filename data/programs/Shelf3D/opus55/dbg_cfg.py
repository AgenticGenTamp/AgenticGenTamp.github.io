import sys, numpy as np, json
import approach, cfgs
from env_client import make_env
cfg=getattr(cfgs, sys.argv[1])
for k,v in cfg.items(): setattr(approach,k,v)
env = make_env()
obs, info = env.reset(seed=int(sys.argv[2]), options={'object_count': int(sys.argv[3])})
ap = approach.GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
last=None
for t in range(int(sys.argv[4])):
    a = ap.get_action(obs); obs, r, term, trunc, info = env.step(a)
    robot, cubes, cup = ap._parse(obs)
    if t % int(sys.argv[5]) == 0 or ap.phase != last:
        c = cubes[ap.cur]
        print(t, ap.phase, 'base', robot[:3].round(2), 'cube', c[:3].round(3), 'q', robot[3:10].round(2), 'qerr', np.abs(approach.wrap(ap.q_t-robot[3:10])).round(2))
    last = ap.phase
