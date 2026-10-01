import numpy as np, sys, time
from env_client import make_env
from approach import GeneratedApproach
seeds=[int(s) for s in sys.argv[1:]]
env=make_env(); res=[]
for sd in seeds:
    t0=time.time()
    obs,info=env.reset(seed=sd)
    ap=GeneratedApproach(env.action_space,env.observation_space,{})
    ap.reset(obs,info)
    R=0; term=False
    for t in range(env.max_steps):
        obs,rew,term,trunc,_=env.step(ap.get_action(obs)); R+=rew
        if term or trunc: break
    res.append((sd,t+1,term,round(time.time()-t0,1)))
    print(f"seed {sd}: steps={t+1} term={term} wall={time.time()-t0:.1f}s nspare={info['object_count']-1}",flush=True)
ok=[r for r in res if r[2]]
print("SOLVED",len(ok),"/",len(res),"mean steps",np.mean([r[1] for r in ok]) if ok else None,
      "max wall",max(r[3] for r in res))
env.close()
