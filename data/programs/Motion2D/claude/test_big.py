import sys, time
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
n=int(sys.argv[1]); off=int(sys.argv[2])
env = make_env()
ap = GeneratedApproach(env.action_space, env.observation_space, {})
fails=[]; steps_all=[]; worst=0
for seed in range(off, off+n):
    obs, info = env.reset(seed=seed)
    t0=time.time(); ap.reset(obs, info); tot=0; term=False
    for i in range(env.max_steps):
        obs, rew, term, trunc, info = env.step(ap.get_action(obs)); tot+=1
        if term or trunc: break
    el=time.time()-t0
    worst=max(worst,el)
    steps_all.append(tot)
    if not term: fails.append(seed); print("FAIL",seed,flush=True)
    if el>20: print("SLOW",seed,round(el,1),flush=True)
print("range",off,off+n,"FAILS",fails,"mean",round(float(np.mean(steps_all)),1),"max",max(steps_all),"worst_time",round(worst,1))
env.close()
