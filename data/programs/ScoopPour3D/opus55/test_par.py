import sys, time, numpy as np
from multiprocessing import Pool
from env_client import make_env
from approach import GeneratedApproach
def run(seed, max_steps=1000):
    try:
        return run_(seed, max_steps)
    except Exception as e:
        return {'seed': seed, 'error': repr(e)[:100]}

def run_(seed, max_steps=1000):
    env = make_env()
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    ph = {}; fails = 0; prev=None; picks=0; tagent=0
    for t in range(max_steps):
        t0=time.time(); a = ap.get_action(obs); tagent+=time.time()-t0
        ph[ap.phase] = ph.get(ap.phase, 0) + 1
        if prev=='lift' and ap.phase=='select': fails+=1
        if prev=='lift' and ap.phase=='carry': picks+=1
        prev=ap.phase
        obs, r, term, trunc, info = env.step(a)
        if term or trunc: break
    g = obs.get_object_from_name('bin_green_0'); gx, gy = obs.get(g,'x'), obs.get(g,'y')
    n_in = n = 0
    for nm in obs.get_object_names():
        if nm.startswith('cube'):
            o = obs.get_object_from_name(nm); n += 1
            if abs(obs.get(o,'x')-gx) < 0.215 and abs(obs.get(o,'y')-gy) < 0.14: n_in += 1
    env.close()
    return dict(seed=seed, steps=t+1, term=term, in_green=n_in, n=n, picks=picks, fails=fails, agent_s=round(tagent,1), phases=ph)
if __name__ == '__main__':
    seeds = [int(s) for s in sys.argv[1:]]
    with Pool(min(7, len(seeds))) as p:
        for r in p.imap_unordered(run, seeds): print(r, flush=True)
