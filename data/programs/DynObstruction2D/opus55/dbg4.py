import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); ph=sys.argv[2]
env = make_env()
ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=seed); ap.reset(obs, info)
R=obs.get_object_from_name('robot')
for t in range(400):
    a = ap.get_action(obs)
    p=ap.phase
    obs, r, term, trunc, info = env.step(np.asarray(a, dtype=float))
    if p==ph:
        print(t, 'rob y %.3f'%obs.get(R,'y'), ' '.join('%s(%.3f,%.3f,%.2f)'%(o.name[:5]+o.name[-1],obs.get(o,'x'),obs.get(o,'y'),obs.get(o,'theta')) for o in sorted(obs.data,key=lambda o:o.name) if o.name not in('robot','target_surface')))
    elif p!=ph and t>0 and 'printed' in dir(): break
