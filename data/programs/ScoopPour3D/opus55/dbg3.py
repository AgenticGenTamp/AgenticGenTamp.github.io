import numpy as np, kin
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=0)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(135):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    if t>=110 and t%3==0:
        base,q=ap._robot(obs)
        print(t, ap.phase, 'qint',ap.q_int.round(3), 'q',q.round(3), 'seed', ap.q_seed.round(3))
