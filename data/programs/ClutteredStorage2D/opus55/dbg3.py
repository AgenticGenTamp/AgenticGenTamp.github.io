import sys
from env_client import make_env
from approach import GeneratedApproach
env = make_env()
obs, info = env.reset(seed=0)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for t in range(200):
    a = ap.get_action(obs)
    if ap.phase in ('extend','lift'): print(t, ap.task, ap.phase, a.round(4), round(ap.rx,3), round(ap.ry,3), round(ap.rth,4), round(ap.arm,3))
    obs, r, term, trunc, info = env.step(a)
