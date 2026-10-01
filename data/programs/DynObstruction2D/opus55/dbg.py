import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 300
env = make_env()
ap = GeneratedApproach(env.action_space, env.observation_space, {})
obs, info = env.reset(seed=seed); ap.reset(obs, info)
R=obs.get_object_from_name('robot')
last=None
for t in range(N):
    a = ap.get_action(obs)
    obs, r, term, trunc, info = env.step(np.asarray(a, dtype=float))
    rob=[round(obs.get(R,f),3) for f in ['x','y','theta','arm_joint','finger_gap']]
    if ap.phase!=last or t%20==0:
        print(t, ap.phase, rob, np.round(a,3), ap.task)
        last=ap.phase
    if term: print('TERM', t); break
for o in sorted(obs.data,key=lambda o:o.name):
    if o.name!='robot': print(o.name, [round(obs.get(o,f),3) for f in ['x','y','theta','width','height']])
