import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
sd=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=sd)
ap=GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs,info)
print("plan", None if ap.plan is None else {k:(np.round(v,3) if isinstance(v,np.ndarray) else v) for k,v in ap.plan.items()})
print("phase",ap.phase)
print("M",obs[20:22],"T",obs[29:31],"hook",obs[9:12],"robot",obs[:3])
last=ap.phase
for i in range(env.max_steps):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    if ap.phase!=last:
        print(f"  step {i} -> {ap.phase} robot={np.round(obs[:3],3)} hook={np.round(obs[9:12],3)} M={np.round(obs[20:22],3)}")
        last=ap.phase
    if i%100==0: print(f"  t={i} ph={ap.phase} stuck={ap.stuck} robot={np.round(obs[:3],3)} q={len(ap.queue)}")
    if term or trunc: print("DONE",i,term); break
print("final dist",np.linalg.norm(obs[20:22]-obs[29:31]),"phase",ap.phase)
np.save(f"dbg{sd}.npy",obs)
env.close()
