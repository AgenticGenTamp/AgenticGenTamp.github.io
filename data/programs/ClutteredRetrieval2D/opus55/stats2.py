import sys, time, multiprocessing as mp, os
import numpy as np
def run(seed):
    from env_client import make_env
    import approach
    for kv in filter(None, os.environ.get('OVR', '').split(',')):
        k, v = kv.split('='); setattr(approach, k, float(v))
    env = make_env(); obs, info = env.reset(seed=seed)
    ap = approach.GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    log = []
    orig = ap._go_grasp
    def gg(path, name, carry=None):
        log.append((name, carry is not None)); return orig(path, name, carry)
    ap._go_grasp = gg
    for steps in range(1, 1001):
        obs, r, term, trunc, info = env.step(ap.get_action(obs))
        if term: break
    return seed, info.get('object_count'), term, steps, log
if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2])
    with mp.Pool(24) as p:
        res = p.map(run, range(a, b))
    for s, n, t, st, log in res:
        if n == 10:
            print(s, st, len(log), sum(1 for x in log if x[0]=='target_block'), sum(1 for x in log if not x[1]), len(set(x[0] for x in log)))
