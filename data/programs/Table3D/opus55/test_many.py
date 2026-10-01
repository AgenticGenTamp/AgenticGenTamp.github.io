import sys, time
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
cnt=int(sys.argv[1]); seeds=range(int(sys.argv[2]),int(sys.argv[3]))
env=make_env(); succ=0
for s in seeds:
    try:
        obs, info = env.reset(seed=s, options={'object_count':cnt})
    except Exception as e:
        print('reset fail',e); break
    ap=GeneratedApproach(env.action_space, env.observation_space, {})
    t0=time.time(); ap.reset(obs, info); rt=time.time()-t0; ok=False
    for t in range(env.max_steps):
        obs,r,te,tr,info=env.step(ap.get_action(obs))
        if te: ok=True; break
        if tr: break
    succ+=ok
    print(f'cnt {cnt} seed {s} n={len([n for n in obs.get_object_names() if n.startswith("cube")])} ok={ok} steps={t+1} reset={rt:.2f} total={time.time()-t0:.1f}',flush=True)
print('success',succ)
env.close()
