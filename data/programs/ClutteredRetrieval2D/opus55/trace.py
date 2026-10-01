import sys, numpy as np
from env_client import make_env
from approach import *
seed = int(sys.argv[1])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for step in range(1000):
    a = ap.get_action(obs)
    qb = ap.q.copy(); held = ap.held
    obs, r, term, trunc, info = env.step(a)
    if term: print("solved", step); break
    ap._parse(obs)
    dq = ap.q - qb; dq[2] = wrap(dq[2])
    exp = a[:4].astype(float).copy(); 
    exp[3] = np.clip(qb[3]+exp[3], 0.1, 0.2) - qb[3]
    if np.abs(dq - exp).max() > 1e-4:
        print(step, "held", held, "a", np.round(a,4), "dq", np.round(dq,4), "qb", np.round(qb,4))
