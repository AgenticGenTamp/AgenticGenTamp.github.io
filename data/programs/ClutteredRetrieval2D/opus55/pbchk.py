import sys, numpy as np
from env_client import make_env
import approach
from approach import *
seed=int(sys.argv[1]); upto=int(sys.argv[2])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for step in range(upto):
    obs, *_ = env.step(ap.get_action(obs))
ap._parse(obs); ap.held=None; ap.compute_soft_now()
cands = ap.grasp_candidates('target_block')
print("ncands", len(cands))
good, pb = ap.place_blockers(cands)
print("good", len(good), "pb", pb)
obs_o, onames = ap.make_obstacles(exclude=('target_block',))
for q in cands[:2]:
    hl = ap.local_of('target_block', q); model = Model(obs_o, hl)
    P = ap.place_configs(hl, q[3])
    wb = model.wall_bad(P); H = model.per_obj(P)
    for k in range(0, len(P), 6):
        print(np.round(P[k],2), "wall", wb[k], [onames[i] for i in np.nonzero(H[k])[0]])
