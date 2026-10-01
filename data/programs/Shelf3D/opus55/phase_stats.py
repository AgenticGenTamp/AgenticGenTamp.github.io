import sys, collections
from env_client import make_env
import approach, cfgs
from approach import GeneratedApproach
if len(sys.argv) > 3:
    for k, v in getattr(cfgs, sys.argv[3]).items(): setattr(approach, k, v)
env = make_env()
obs, info = env.reset(seed=int(sys.argv[1]), options={'object_count': int(sys.argv[2])})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
cnt = collections.Counter(); n=0; seq=[]
while n < 1000:
    a = ap.get_action(obs); cnt[ap.phase] += 1
    if not seq or seq[-1][0] != ap.phase: seq.append([ap.phase, 0])
    seq[-1][1] += 1
    obs, r, term, trunc, info = env.step(a); n += 1
    if term: break
print(sys.argv[1:], 'steps', n, 'term', term, dict(cnt))
print(' '.join(f'{p[:3]}{c}' for p, c in seq)[:1500])
