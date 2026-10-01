import sys, time
import numpy as np
from env_client import make_env
from approach_fast import GeneratedApproach
seeds = [int(s) for s in sys.argv[1].split(',')] if len(sys.argv) > 1 else list(range(10))
nobj = int(sys.argv[2]) if len(sys.argv) > 2 else None
env = make_env()
succ = 0
for seed in seeds:
    kw = {} if nobj is None else {'options': {'object_count': nobj}}
    try:
        obs, info = env.reset(seed=seed, **kw)
    except TypeError:
        obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    t0 = time.time(); ret = 0; ok = False; tpol = 0
    for t in range(env.max_steps):
        t1 = time.time(); a = ap.get_action(obs); tpol += time.time() - t1
        obs, r, term, trunc, info = env.step(a)
        ret += r
        if term: ok = True; break
        if trunc: break
    succ += ok
    print(f"seed {seed} n={info.get('object_count')} ok={ok} steps={t+1} ret={ret:.1f} pol_time={tpol:.1f}s wall={time.time()-t0:.1f}s", flush=True)
print(f"SUCCESS {succ}/{len(seeds)}")
env.close()
