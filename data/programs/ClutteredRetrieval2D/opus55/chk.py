import sys, numpy as np
from env_client import make_env
from approach import *
env = make_env(); obs, info = env.reset(seed=int(sys.argv[1]))
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
name = sys.argv[2]
obs_all, names = ap.make_obstacles()
model = Model(obs_all, None)
others, onames = ap.make_obstacles(exclude=(name,))
for q in ap.grasp_configs(name):
    hits = [names[i] for i in np.nonzero(model.per_obj(q[None])[0])[0]]
    gp = model.body_polys(q[None])[0]
    prox = [onames[i] for i in np.nonzero(others.poly_hit(gp, 0.022, per_obj=True)[0])[0]]
    print(np.round(q,3), "hits", hits, "wall", model.wall_bad(q[None])[0], "prox", prox)
c = ap.grasp_candidates(name); print("n cands", len(c))
for q in c:
    hl = ap.local_of(name, q)
    print("retreat", ap.retreat_ok(name, q), "dump", ap.plan_dump(1.0, q, hl, name) is not None)
