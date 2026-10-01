import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 300
env = make_env()
ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=seed); ap.reset(obs, info)
R=obs.get_object_from_name('robot')
def ob(obs): return ' '.join('%s(%.2f,%.2f,%.2f)'%(o.name[:5]+o.name[-1],obs.get(o,'x'),obs.get(o,'y'),obs.get(o,'theta')) for o in sorted(obs.data,key=lambda o:o.name) if o.name!='robot')
print('start', ob(obs))
last=None
for t in range(N):
    a = ap.get_action(obs)
    obs, r, term, trunc, info = env.step(np.asarray(a, dtype=float))
    if ap.phase!=last:
        tk=ap.task or {}
        print(t, ap.phase, tk.get('kind'), tk.get('name'), '| ', ob(obs))
        last=ap.phase
    if term: print('TERM', t); break
