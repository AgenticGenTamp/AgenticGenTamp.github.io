import numpy as np, sys, kin
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); cnt=int(sys.argv[2])
env = make_env(); obs, info = env.reset(seed=seed, options={'object_count':cnt})
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
last=None; t0=0
for t in range(1000):
    a = ap.get_action(obs)
    key=(ap.phase,ap.target)
    if key!=last:
        c=ap.cubes.get(ap.target)
        inb=sum(ap._in_box(v,m=0) for v in ap.cubes.values())
        print(f"{t0:4d}-{t:4d} {last} -> {key} th={ap.th if ap.th is None else round(ap.th,2)} base={ap.base.round(2)} tip={ap._tip_xy().round(2)} cube={None if c is None else c[:3].round(3)} dq={np.abs(ap.q-ap.qh_t).max():.2f} inbox={inb}")
        last=key; t0=t
    obs, r, term, trunc, info = env.step(a)
    if term: print('TERM',t); break
