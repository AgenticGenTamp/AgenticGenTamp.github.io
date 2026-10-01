import sys, collections
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
        cnt[k] += 1
        obs, r, term, trunc, info = env.step(a)
        if term: break
    env.close()
for k, v in cnt.most_common(): print(k, v)
