import sys, numpy as np
from env_client import make_env
import approach
from approach import *
seed=int(sys.argv[1]); upto=int(sys.argv[2]); name=sys.argv[3]
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for step in range(upto):
    obs, *_ = env.step(ap.get_action(obs))
ap._parse(obs)
obs_all, names = ap.make_obstacles()
model = Model(obs_all, None)
others, onames = ap.make_obstacles(exclude=(name,))
for q in ap.grasp_configs(name):
    if model.wall_bad(q[None])[0]: continue
    hits = [names[i] for i in np.nonzero(model.per_obj(q[None])[0])[0]]
    gp = model.body_polys(q[None])[0]
    prox = [onames[i] for i in np.nonzero(others.poly_hit(gp, 0.022, per_obj=True)[0])[0]]
    print(np.round(q,3), "hits", hits, "prox", prox)
for n in ap.rects: print(n, ap.center(n).round(2).tolist())
