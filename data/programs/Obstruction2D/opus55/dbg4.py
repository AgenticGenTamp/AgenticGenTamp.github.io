import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed, count, T = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
env = make_env()
obs, info = env.reset(seed=seed, options={'object_count': count})
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for t in range(T):
    a = ap.get_action(obs)
    obs, r, term, trunc, info = env.step(a)
ap._parse(obs)
print('robot', {k:round(v,3) for k,v in ap.robot.items()})
print('dest', ap.block_dest, ap.block_grel, 'surface', ap.surface)
for n,b in ap._all_movable().items(): print(n, {k:round(v,3) for k,v in b.items() if k!='name'}, 'hit', ap._column_hit(b))
print(ap.plan)
