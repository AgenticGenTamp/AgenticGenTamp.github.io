import sys
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
obs,info=env.reset(seed=int(sys.argv[1]))
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
ap._read(obs)
start=(ap.rx,ap.ry)
perm=tuple(ap.buttons)
for o in [("below","fast"),("left","fast"),("right","fast"),("left","reach"),("right","reach")]+ap._grasp_options(start):
    ap._first_opt=None
    c=ap._seq_cost(start, ap.rth, perm, 0, o, trace=[])
    print(o, round(c,2), ap._first_opt)
