import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
import robot
from s2 import *
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
print("plan:")
for it in ap.plan: print("  ",it[0], np.round(it[1],4) if isinstance(it[1],np.ndarray) else it[1])
lastphase=-1
for i in range(env.max_steps):
    a=ap.get_action(obs)
    if ap.phase!=lastphase:
        T=ap.  _tool(obs)
        print(i,"phase",ap.phase,"tool",np.round(T[:3,3],4),"grasp",rinfo(obs)['grasp_active'],flush=True)
        lastphase=ap.phase
    obs,r,term,trunc,info=env.step(a)
    if term or trunc: print("TERM",i); break
print("final block",np.round(opos(obs,'target_block'),4),"region",np.round(opos(obs,'target_region'),4),"grasp",rinfo(obs)['grasp_active'])
