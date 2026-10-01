import time, sys, numpy as np
from env_client import make_env
import approach
seeds=[int(x) for x in sys.argv[1:]] or [0,1,2]
env=make_env()
ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
for seed in seeds:
    obs,info=env.reset(seed=seed)
    t=time.time(); ap.reset(obs,info); print("seed",seed,"plan time",round(time.time()-t,2))
    for k in range(2):
        print(" rover",k,[(t2['kind'],t2['obj'],np.round(ap.graph.pts[t2['node']],2)) for t2 in ap.tasks[k]])
env.close()
