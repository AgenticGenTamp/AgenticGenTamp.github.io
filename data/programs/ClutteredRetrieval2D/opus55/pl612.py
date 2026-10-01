import sys, numpy as np
from env_client import make_env
import approach
from approach import *
env = make_env(); obs, info = env.reset(seed=int(sys.argv[1]))
for sl in (0.0, 0.026):
    approach.PLACE_SLACK = sl
    ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info); ap.compute_soft_now()
    cands = ap.grasp_candidates('target_block'); good, pb = ap.place_blockers(cands)
    res=[]
    for qg in good[:4]:
        hl = ap.local_of('target_block', qg)
        approach._ITERS[0]=0
        p = ap.plan_place(2.0, qg, hl)
        obs_o,_ = ap.make_obstacles(exclude=('target_block',)); m = Model(obs_o, hl)
        P = ap.place_configs(hl, qg[3], qg)
        res.append((p is not None, approach._ITERS[0], int((~m.hits(P)).sum())))
    print(sl, len(good), res)
qg = good[0]; hl = ap.local_of('target_block', qg)
for sl in (0.0, 0.026):
    approach.PLACE_SLACK = sl
    P = ap.place_configs(hl, qg[3], qg); print(sl, np.round(P[:3],3).tolist(), m.wall_bad(P[:3]))
print(ap.center('target_region'), ap.rects['target_region'])
obs_o, on = ap.make_obstacles(exclude=('target_block',)); m = Model(obs_o, hl)
for sl in (0.0, 0.026):
    approach.PLACE_SLACK = sl
    P = ap.place_configs(hl, qg[3], qg)
    H = m.per_obj(P); print(sl, [[on[i] for i in np.nonzero(h)[0]] for h in H[:6]], m.hits(P[:6]))
print(on)
