import sys, numpy as np
from env_client import make_env
from approach import *
env = make_env()
obs, info = env.reset(seed=int(sys.argv[1]), options={'object_count': int(sys.argv[2])})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(int(sys.argv[3])):
    a = ap.get_action(obs); obs, r, term, trunc, info = env.step(a)
    robot, cubes, cup = ap._parse(obs)
    if ap.phase in ('insert','release') :
        c = cubes[ap.cur]
        print(t, ap.phase, 'base', robot[:3].round(3), 'cube', c[:3].round(3), 'qerr', np.abs(wrap(ap.q_t-robot[3:10])).max().round(3))
