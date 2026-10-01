import sys
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
obs,info=env.reset(seed=159)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for n in range(34):
    a=ap.get_action(obs); obs,*_=env.step(a)
    if n>=26: print(n, [round(x,3) for x in a], round(ap.rx,3), round(ap.ry,3), round(ap.rth,3), ap.plan, ap.cur_target)
ap._read(obs)
start=(ap.rx,ap.ry)
btns=ap.buttons
print(btns, ap.loc)
import itertools
for perm in itertools.permutations(btns):
    print([b[0] for b in perm], ap._seq_cost(start, ap.rth, perm, None))
for b in btns:
    print(b[0], ap._button_target(b, start), ap._stick_target(b,start))
