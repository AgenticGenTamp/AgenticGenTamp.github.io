import sys, numpy as np, time
from env_client import make_env
import approach as A
lo,hi=int(sys.argv[1]),int(sys.argv[2])
oc = int(sys.argv[3]) if len(sys.argv)>3 else None
env = make_env()
ap = A.GeneratedApproach(env.action_space, env.observation_space, {})
fails=[]; steps=[]; rep=[]; slow=[]
for seed in range(lo,hi):
    obs, info = env.reset(seed=seed) if oc is None else env.reset(seed=seed, options={'object_count':oc})
    ap.reset(obs, info)
    t0=time.time(); term=False; t=0
    for t in range(env.max_steps):
        try: a = ap.get_action(obs)
        except Exception as e:
            print("EXC seed",seed,repr(e)[:200]); break
        obs, r, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
        if term or trunc: break
    dt=time.time()-t0
    if term: steps.append(t+1)
    else: fails.append((seed, info.get('object_count')))
    if ap.replans: rep.append((seed,ap.replans))
    if dt>15: slow.append((seed,round(dt,1)))
print("oc",oc,"solved",len(steps),"/",hi-lo,"mean %.1f"%(np.mean(steps) if steps else -1),"max",max(steps) if steps else None)
print(" fails",fails[:15]," replans",rep[:15], " slow",slow[:5])
env.close()
