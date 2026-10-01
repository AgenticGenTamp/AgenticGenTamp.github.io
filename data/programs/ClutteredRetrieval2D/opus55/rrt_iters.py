import sys, numpy as np, approach
from env_client import make_env
from approach import *
env = make_env(); obs, info = env.reset(seed=int(sys.argv[1]))
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
name = sys.argv[2]
ap.compute_soft_now()
obs_all, _ = ap.make_obstacles(soft=(name,))
model = Model(obs_all, None)
for q in ap.grasp_candidates(name)[:6]:
    res = []
    for trial in range(5):
        approach._ITERS[0] = 0
        p = rrt_connect(model, ap.q, q, np.random.default_rng(trial), time_limit=20)
        res.append(approach._ITERS[0] if p is not None else -1)
    print(np.round(q,2), res)
q = ap.grasp_candidates(name)[1]
t = Tree(q); seed_retreat(model, t); print("goal retreat nodes", t.n, t.nodes[t.n-1].round(3))
t = Tree(ap.q); seed_retreat(model, t); print("start retreat nodes", t.n)
# free fraction around goal
rng = np.random.default_rng(0)
for rad in [0.1, 0.2, 0.3]:
    Q = q[None] + np.column_stack([rng.uniform(-rad,rad,2000), rng.uniform(-rad,rad,2000), rng.uniform(-np.pi,np.pi,2000), rng.uniform(-0.1,0.1,2000)])
    Q[:,3] = np.clip(Q[:,3],0.1,0.2)
    print(rad, "free frac", (~model.hits(Q)).mean())
for st in [0.06, 0.2, 0.3]:
    res=[]
    for trial in range(5):
        approach._ITERS[0] = 0
        p = rrt_connect(model, ap.q, q, np.random.default_rng(trial), time_limit=20, step=st)
        res.append(approach._ITERS[0] if p is not None else -1)
    print("step", st, res)
