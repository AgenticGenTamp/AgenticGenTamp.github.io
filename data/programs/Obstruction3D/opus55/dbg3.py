import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
from kin import fk
seed=int(sys.argv[1]); oc=int(sys.argv[2]) if len(sys.argv)>2 else None
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':oc} if oc is not None else None)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for n,(p,he) in ap.objs.items(): print(n, p.round(3), he.round(3))
for t in range(int(__import__("os").environ.get("NS","400"))):
    a = ap.get_action(obs); obs, r, te, *_ = env.step(a)
    tool = fk(ap.base, ap.q)[:3,3]
    print(t, ap.phase[:6], (ap.task or {}).get('name','')[-6:], tool.round(3), 'sc', ap.scale, 'g', int(ap.grasped), 'gr', a[10], 'np', len(ap.path), 'dq', np.abs(a[3:10]).max().round(2))
    if te: print('DONE', t+1); break
