import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env()
obs, info = env.reset(seed=1)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for t in range(400):
    a = ap.get_action(obs)
    if ap.task and ap.task[0]=='fetch' and ap.phase=='retreat' or (ap.task is None and ap.phase=='retreat'):
        break
    obs, r, term, trunc, info = env.step(a)
ap._parse(obs)
print('t', t, 'arm', ap.arm, 'vac', ap.vac)
before = {n: b['poly'][:,1].min() for n,b in ap.blocks.items()}
obs, *_ = env.step(np.array([0,0,0,-0.1,0], dtype=np.float32))
ap._parse(obs)
print('arm', ap.arm, 'vac', ap.vac)
for n,b in ap.blocks.items(): print(n, round(before[n],4), round(b['poly'][:,1].min(),4))
