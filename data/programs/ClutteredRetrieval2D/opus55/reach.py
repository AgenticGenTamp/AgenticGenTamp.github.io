import sys, numpy as np, time
from scipy import ndimage
from env_client import make_env
import approach; approach.DEBUG=False
from approach import *
seed=int(sys.argv[1]); upto=int(sys.argv[2]); name=sys.argv[3]
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for step in range(upto):
    obs, *_ = env.step(ap.get_action(obs))
ap._parse(obs); ap.held=None; ap.queue=[]; ap.compute_soft_now()
o,_ = ap.make_obstacles(soft=(name,))
xs, clr = clearance_grid(o)
lab,_ = ndimage.label(clr >= 0.104)
cell=lambda p:(int(round((p[0]-xs[0])/GRID_RES)), int(round((p[1]-xs[0])/GRID_RES)))
print("robot", ap.q.round(3), "label", lab[cell(ap.q)], "clr", clr[cell(ap.q)].round(3))
model = Model(o, None)
for qg in ap.grasp_candidates(name)[:19:3]:
    approach._ITERS[0]=0
    p = rrt_connect(model, ap.q, qg, np.random.default_rng(0), time_limit=5)
    print(np.round(qg,2), "label", lab[cell(qg)], "clr", clr[cell(qg)].round(3), "rrt", p is not None, approach._ITERS[0])
