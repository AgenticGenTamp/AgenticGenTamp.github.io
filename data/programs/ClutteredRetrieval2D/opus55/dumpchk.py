import sys, numpy as np, time
from env_client import make_env
import approach; approach.DEBUG=False
from approach import *
seed=int(sys.argv[1]); upto=int(sys.argv[2]); names=sys.argv[3].split(',')
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for step in range(upto):
    obs, *_ = env.step(ap.get_action(obs))
ap._parse(obs); ap.held=None; ap.queue=[]; ap.compute_soft_now()
print("q", ap.q.round(3))
for name in names:
    cands = ap.grasp_candidates(name)
    n=0; res=[]
    t=time.time()
    for path in ap.iter_grasp_paths(name, 5.0, maxc=6):
        qg = path[-1]; hl = ap.local_of(name, qg)
        approach._ITERS[0]=0
        carry = ap.plan_dump(5.0, qg, hl, name)
        res.append((np.round(qg,2).tolist(), carry is not None, approach._ITERS[0]))
        n+=1
        if n>=3: break
    print(name, "ncands", len(cands), "time %.1f"%(time.time()-t), res)
