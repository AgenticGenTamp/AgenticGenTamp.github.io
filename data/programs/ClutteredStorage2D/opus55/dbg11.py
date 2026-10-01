import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); T0=int(sys.argv[2]); T1=int(sys.argv[3])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
for t in range(T1):
    a=ap.get_action(obs)
    if t>=T0: print(t, ap.task, ap.phase, 'r', round(ap.rx,3), round(ap.ry,3), round(ap.rth,3), round(ap.arm,3), 'vac', ap.vac, 'a', np.round(a,3), 'rn', getattr(ap,'retreat_n',None), 'stuck', ap.stuck)
    obs,*_=env.step(a)
ap._parse(obs)
print('plan', ap._plan_push(), 'lazy', ap.lazy_cols, 'tot', ap.plan_total, 'failed', ap.failed)
print('choose', ap._choose_task())
for n,b in ap.blocks.items(): print(n, np.round(b['poly'],3).tolist(), 'in' if ap._inside(b) else '')
print('sx1', ap.sx1, ap.sw1)
ap.allowed_cols=None
for n in ('block5','block6'):
    print(n, len(ap._grasp_options(n)))
