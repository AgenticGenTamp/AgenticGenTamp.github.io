import sys, numpy as np, time
from env_client import make_env
import approach as A
lo,hi=int(sys.argv[1]),int(sys.argv[2])
ocs=[int(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else [None]
env = make_env(); ap = A.GeneratedApproach(env.action_space, env.observation_space, {})
for oc in ocs:
    fails=[]; steps=[]; rep=[]; slow=[]
    for seed in range(lo,hi):
        obs, info = env.reset(seed=seed) if oc is None else env.reset(seed=seed, options={'object_count':oc})
        ap.reset(obs, info); t0=time.time(); term=False; t=0
        for t in range(env.max_steps):
            try: a = ap.get_action(obs)
            except Exception as e:
                print("EXC seed",seed,oc,repr(e)[:300]); break
            obs, r, term, trunc, info = env.step(a)
            if term or trunc: break
        if term: steps.append(t+1)
        else: fails.append(seed)
        if ap.replans: rep.append((seed,ap.replans))
        if time.time()-t0>10: slow.append((seed,round(time.time()-t0,1)))
    print("oc",oc,"solved",len(steps),"/",hi-lo,"mean %.1f"%(np.mean(steps) if steps else -1),
          "max",max(steps) if steps else None,"fails",fails[:10],"replans",rep[:10],"slow",slow[:6], flush=True)
env.close()
