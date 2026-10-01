import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env = make_env(); obs, info = env.reset(seed=0)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for t in range(200):
    a = ap.get_action(obs)
    if t%10==0:
        bt=ap._pick_base(ap.target, ap.th) if ap.target else None
        print(t, ap.phase, ap.base.round(3), bt.round(3) if bt is not None else None, np.abs(kin_q:=ap.q-ap.q_hover).max().round(3), a[:3].round(3))
    obs, r, term, trunc, info = env.step(a)
