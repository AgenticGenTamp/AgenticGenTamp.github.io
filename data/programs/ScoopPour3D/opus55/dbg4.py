import numpy as np, kin
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=0)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(130):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    if 86<=t<=126 and t%2==0:
        base,q=ap._robot(obs)
        print(t, ap.phase, 'err', round(ap.tool_err,4), 'I', ap.I.round(3), 'q-qint', (q-ap.q_int).round(3), base.round(3))
