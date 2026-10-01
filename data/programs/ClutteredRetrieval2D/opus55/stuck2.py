import sys, numpy as np
from env_client import make_env
import approach; approach.DEBUG=False
from approach import *
seed=int(sys.argv[1]); upto=int(sys.argv[2]); hname=sys.argv[3]
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for step in range(upto):
    obs, *_ = env.step(ap.get_action(obs))
ap._parse(obs)
ap.held=hname; ap.held_local=ap.local_of(hname, ap.q); ap.compute_soft_now()
print("q", ap.q.round(4), "soft", ap.soft_now)
names = ap.obstacle_names(exclude=(hname,))
obs_all,_ = ap.make_obstacles(exclude=(hname,))
for mm in [0.0, 0.004]:
    m = Model(obs_all, ap.held_local, mm)
    print(mm, "free", m.free(ap.q), "wall", m.wall_bad(ap.q[None]), [ (k, obs_all.poly_hit(P, mm)) for k,P in enumerate(m.body_polys(ap.q[None]))], obs_all.circle_hit(ap.q[None,:2], BASE_R, mm))
    print([names[i] for i in np.nonzero(m.per_obj(ap.q[None], mm)[0])[0]])
print("held corners", ap.corners(hname).round(3))
import time
for n in ap.rects: print(n, ap.center(n).round(2))
t=time.time(); approach._ITERS[0]=0
p = ap.plan_place(20.0, ap.q, ap.held_local)
print("place budget20", p is not None, approach._ITERS[0], time.time()-t)
