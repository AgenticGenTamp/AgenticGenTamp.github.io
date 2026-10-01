import sys, numpy as np, approach
from env_client import make_env
from approach import *
env = make_env(); obs, info = env.reset(seed=int(sys.argv[1]))
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
name = sys.argv[2]
ap.compute_soft_now()
obs_all, _ = ap.make_obstacles(soft=(name,))
model = Model(obs_all, None)
q = ap.grasp_candidates(name)[0]
ta = Tree(ap.q); seed_retreat(model, ta); tb = Tree(q); seed_retreat(model, tb)
ia, qa = retract_branch(model, ta); ib, qb = retract_branch(model, tb)
print(ia, qa, ib, qb)
xs, clr = clearance_grid(model.obs); R = body_radius(model)+model.m+0.005
print("R", R, "free frac", (clr>=R).mean())
if qa is not None and qb is not None:
    i=lambda p:(int(round((p[0]-xs[0])/GRID_RES)),int(round((p[1]-xs[0])/GRID_RES)))
    print("clr at a", clr[i(qa)], "at b", clr[i(qb)])
    print(grid_path(model, qa, qb))
import heapq
m=model.m; free = clr >= BASE_R+m+0.002
gi=i(qb); print("goal free", free[gi], clr[gi], BASE_R+m+0.002)
print(np.round(clr[gi[0]-3:gi[0]+4, gi[1]-3:gi[1]+4],3))
