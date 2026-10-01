import sys, collections, numpy as np
from env_client import make_env
from approach import GeneratedApproach
cnt = collections.Counter()
for seed in range(int(sys.argv[1]), int(sys.argv[2])):
    env = make_env()
    obs, info = env.reset(seed=seed)
    ap = GeneratedApproach(env.action_space, env.observation_space, {})
    ap.reset(obs, info)
    for t in range(1000):
        a = ap.get_action(obs)
        k = (ap.task[0] if ap.task else None, ap.phase)
        mv = abs(a[0])>1e-6 or abs(a[1])>1e-6
        rot = abs(a[2])>1e-6
        typ = 'move+rot' if mv and rot else 'move' if mv else 'rot' if rot else 'arm' if abs(a[3])>1e-6 else 'noop'
        if mv and max(abs(a[0]),abs(a[1]))<0.045: typ += '_short'
        cnt[(k, typ)] += 1
        obs, r, term, trunc, info = env.step(a)
        if term: break
    env.close()
for k, v in cnt.most_common(): print(k, v)
