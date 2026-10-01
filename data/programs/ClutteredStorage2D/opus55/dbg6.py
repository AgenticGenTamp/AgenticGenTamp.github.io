import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed, t0, t1 = map(int, sys.argv[1:4])
env = make_env()
obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for t in range(t1):
    a = ap.get_action(obs)
    if t >= t0:
        b = ap.blocks.get(ap.task[1]) if ap.task else None
        print(t, ap.task, ap.phase, a.round(3), round(ap.rx,3), round(ap.ry,3), round(ap.rth,3), round(ap.arm,3), 'vac', ap.vac, 'goal', np.round(getattr(ap,'goal',0),3), None if b is None else b['center'].round(3))
    obs, r, term, trunc, info = env.step(a)
ap._parse(obs)
b = ap.blocks['block1']
h = np.array([np.cos(ap.rth), np.sin(ap.rth)])
proj = (b['poly'] - np.array([ap.rx, ap.ry])) @ h
print('arm', ap.arm, 'proj', proj.round(4), 'center', b['center'])
