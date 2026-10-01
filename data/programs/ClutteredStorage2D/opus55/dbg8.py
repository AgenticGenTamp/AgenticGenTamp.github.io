import sys, math, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); T=int(sys.argv[2]); name=sys.argv[3]
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(T):
    a=ap.get_action(obs); obs,*_=env.step(a)
ap._parse(obs)
print('robot',ap.rx,ap.ry,ap.rth,ap.arm,'vac',ap.vac)
for n,b in ap.blocks.items():
    print(n, np.round(b['poly'],3).tolist(), round(b['th'],3), 'inside' if ap._inside(b) else '')
pass
print('sx1',ap.sx1,'sw1',ap.sw1,'cols',ap._columns())
print('plan', ap._plan_push(), 'cap_zero', ap.cap_zero)
