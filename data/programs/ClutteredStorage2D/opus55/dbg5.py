import sys
from env_client import make_env
from approach import GeneratedApproach
seed = int(sys.argv[1])
env = make_env()
obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
prev=None
for t in range(1000):
    a = ap.get_action(obs)
    key=(ap.task, ap.phase)
    if key!=prev: print(t, ap.task, ap.phase, round(ap.rx,3), round(ap.ry,3), round(ap.rth,3), round(ap.arm,3))
    prev=key
    obs, r, term, trunc, info = env.step(a)
    if term: print('done', t); break
