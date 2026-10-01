import sys, time, numpy as np
from env_client import make_env
import approach; approach.DEBUG=False
from approach import *
seed = int(sys.argv[1]); upto=int(sys.argv[2])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for step in range(upto):
    a = ap.get_action(obs)
    obs, r, term, trunc, info = env.step(a)
ap._parse(obs); ap.compute_soft_now()
if ap.held is None:
    ap.held="obstruction5"; ap.held_local=ap.local_of("obstruction5", ap.q)
print("held", ap.held, "q", ap.q.round(3), "soft", ap.soft_now)
if ap.held:
    hl = ap.held_local
    approach._ITERS[0]=0
    p = ap.plan_dump(10.0, ap.q, hl, ap.held)
    print("dump with budget 10:", p is not None, approach._ITERS[0])
    obs_all, _ = ap.make_obstacles(exclude=(ap.held,))
    m = Model(obs_all, hl)
    rng=np.random.default_rng(0)
    for rad in [0.02,0.05,0.1]:
        Q = ap.q[None] + np.column_stack([rng.uniform(-rad,rad,(500,2)), rng.uniform(-rad*3,rad*3,500), rng.uniform(-rad,rad,500)])
        Q[:,3]=np.clip(Q[:,3],0.1,0.2)
        print(rad, "free frac", (~m.hits(Q)).mean(), "wallbad", m.wall_bad(Q).mean() if hasattr(m,'wall_bad') else None)
    print("cur free?", m.free(ap.q) if hasattr(m,'free') else None, "margin", m.m)
names = ap.obstacle_names(exclude=(ap.held,))
print(list(zip(names, m.per_obj(ap.q[None])[0] if m.per_obj(ap.q[None]).ndim>1 else m.per_obj(ap.q[None]))))
m0 = Model(obs_all, hl, 0.0); print("free at margin0", m0.free(ap.q))
print("wall", m.wall_bad(ap.q[None]))
for mm in [-0.001,-0.003,-0.01,-0.03]:
    print(mm, Model(obs_all, hl, mm).free(ap.q))
print("obs5", ap.corners("obstruction5").round(4)); print("obs2", ap.corners("obstruction2").round(4))
print("body", [np.round(p,3) for p in m.body_polys(ap.q[None])])
for k,P in enumerate(m.body_polys(ap.q[None])):
    print("poly",k, obs_all.poly_hit(P, 0.0), obs_all.poly_hit(P, -0.01))
print("circle", obs_all.circle_hit(ap.q[None,:2], BASE_R, 0.0))
i2 = names.index("obstruction2"); print("obs2 in obs_all", np.round(obs_all.P[i2],3), obs_all.caps[i2])
