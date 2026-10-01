import numpy as np, kin
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=0)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(215):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    if t>=108 and t%4==0 or t in (60,61,62,63,64,65,66):
        base,q=ap._robot(obs)
        print(t, ap.phase, 'err',(q-ap.q_int).round(3), 'a',np.round(a[3:10],2), 'b',base.round(3))
