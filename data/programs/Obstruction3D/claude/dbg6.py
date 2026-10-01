import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
import robot
from s2 import *
n=int(sys.argv[1]); seed=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed, options={"object_count":n})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
print("tasks",[(t['obj'],np.round(t['dests'][0],3)) for t in ap.tasks])
for nm in sorted(obs.get_object_names()):
    if nm!='robot': print(" ",nm,np.round(opos(obs,nm),4),np.round(half(obs,nm),4))
last=None
for i in range(env.max_steps):
    a=ap.get_action(obs)
    key=(ap.tasks[0]['obj'] if ap.tasks else None, ap.stage, ap.dest_try, ap.retry)
    if key!=last:
        T=ap._tool(obs)
        print(i,key,"tool",np.round(T[:3,3],4),"hold",rinfo(obs)['grasp_active'],flush=True)
        last=key
    obs,r,term,trunc,info=env.step(a)
    if term or trunc: print("TERM",i); break
print("final block",np.round(opos(obs,'target_block'),4),"target",np.round(ap.blk_target,4))
