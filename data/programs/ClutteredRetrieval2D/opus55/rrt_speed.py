import time, numpy as np
from env_client import make_env
from approach import *
env = make_env(); obs, info = env.reset(seed=0)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
obst, names = ap.make_obstacles()
m = Model(obst, None)
rng = np.random.default_rng(0)
Q = np.column_stack([rng.uniform(0.1,2.4,2000), rng.uniform(0.1,2.4,2000), rng.uniform(-3,3,2000), rng.uniform(0.1,0.2,2000)])
t=time.time(); [m.free(q) for q in Q[:500]]; print("free per call us", (time.time()-t)/500*1e6)
t=time.time(); [m.edge_free(Q[i],Q[i+1]) for i in range(200)]; print("edge per call us", (time.time()-t)/200*1e6)
import approach
approach.ITERS_PER_SEC=10**9
t=time.time(); p=rrt_connect(m, ap.q, np.array([0.3,0.3,0,0.1]), rng, time_limit=1.0); print("rrt", time.time()-t, p is not None)
