import sys, numpy as np, time
from env_client import make_env
import approach
from approach import *
seed=int(sys.argv[1]); upto=int(sys.argv[2])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for step in range(upto):
    obs, *_ = env.step(ap.get_action(obs))
ap._parse(obs); ap.compute_soft_now()
print("held", ap.held, ap.q.round(2))
hl = ap.held_local
obs_o,_ = ap.make_obstacles(exclude=('target_block',))
m = Model(obs_o, hl)
for r in range(8):
    ap.rng = np.random.default_rng(r); approach._ITERS[0]=0; t=time.time()
    p = ap.plan_place(4.0, ap.q, hl)
    n = len(resample_path(p, m)) if p is not None else None
    print(r, n, approach._ITERS[0], round(time.time()-t,2))
