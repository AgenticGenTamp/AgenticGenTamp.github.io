import sys, numpy as np, time
from env_client import make_env
import approach as A
lo,hi=int(sys.argv[1]),int(sys.argv[2])
oc = int(sys.argv[3]) if len(sys.argv)>3 else None
env = make_env()
ap = A.GeneratedApproach(env.action_space, env.observation_space, {})
fails=[]; steps=[]
for seed in range(lo,hi):
    kw={'seed':seed}
    obs, info = env.reset(**kw) if oc is None else env.reset(seed=seed, options={'object_count':oc})
    ap.reset(obs, info)
    t0=time.time(); term=False
    for t in range(env.max_steps):
        try:
            a = ap.get_action(obs)
        except Exception as e:
            print("EXC seed",seed,repr(e)); break
        obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        if term or trunc: break
    if term: steps.append(t+1)
    else: fails.append((seed, info.get('object_count'), t+1))
    if term and time.time()-t0>10: print("slow",seed,time.time()-t0)
print("solved",len(steps),"/",hi-lo,"mean",np.mean(steps) if steps else None,"max",max(steps) if steps else None)
print("fails",fails[:20])
env.close()
