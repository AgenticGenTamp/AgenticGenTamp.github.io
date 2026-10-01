import sys, json
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed, count, T = map(int, sys.argv[1:4]); acts = json.loads(sys.argv[4])
env = make_env()
obs, info = env.reset(seed=seed, options={'object_count': count} if count else None)
ap = GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs, info)
for t in range(T):
    obs, *_ = env.step(ap.get_action(obs))
for a in acts:
    obs, *_ = env.step(np.array(a, dtype=np.float32))
    ap._parse(obs); r = ap.robot
    print(a, 'rob', round(r['x'],4), round(r['y'],4), round(r['arm'],4), r['vac'])
for n,b in ap._all_movable().items(): print(n, {k:round(v,3) for k,v in b.items() if k!='name'})
