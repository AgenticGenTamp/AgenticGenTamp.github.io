import sys, numpy as np, time
from env_client import make_env
import approach; approach.DEBUG=False
from approach import *
seed=int(sys.argv[1]); upto=int(sys.argv[2]); B=float(sys.argv[3])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for step in range(upto):
    obs, *_ = env.step(ap.get_action(obs))
ap._parse(obs); ap.held=None; ap.queue=[]; ap.compute_soft_now()
cands = ap.grasp_candidates('target_block')
good, pb = ap.place_blockers(cands)
print("cands", len(cands), "good", len(good), "pblock", pb)
for qg in cands:
    hl = ap.local_of('target_block', qg)
    approach._ITERS[0]=0
    p = ap.plan_place(B, qg, hl)
    print(np.round(qg,2), "retreat_ok", '', p is not None, approach._ITERS[0], ap.escape_blockers('target_block', qg, hl))
