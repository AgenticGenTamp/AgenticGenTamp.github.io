import sys, time, importlib
import numpy as np
from env_client import make_env
from approach import GeneratedApproach

def run(seed, object_count=None, render=False, verbose=False):
    env = make_env()
    kw = {}
    if object_count is not None:
        kw['options'] = {'object_count': object_count}
    try:
        obs, info = env.reset(seed=seed, **kw)
    except Exception as e:
        obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    t0 = time.time()
    total = 0.0
    term = False
    steps = 0
    for i in range(env.max_steps):
        a = ap.get_action(obs)
        obs, r, term, trunc, info = env.step(a)
        total += r; steps += 1
        if term or trunc: break
    dt = time.time()-t0
    env.close()
    return dict(seed=seed, n=info.get('object_count'), term=term, steps=steps, ret=total, wall=round(dt,1))

if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or list(range(5))
    for s in seeds:
        print(run(s), flush=True)
