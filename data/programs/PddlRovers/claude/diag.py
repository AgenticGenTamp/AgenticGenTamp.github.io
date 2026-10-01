import sys, numpy as np
from env_client import make_env
import approach
seed=int(sys.argv[1]) if len(sys.argv)>1 else 0
maxs=int(sys.argv[2]) if len(sys.argv)>2 else 300
env=make_env(); obs,info=env.reset(seed=seed)
ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
ap.reset(obs,info)
for k in range(2):
    print("plan",k,[(t['kind'],t['obj'],np.round(ap.graph.pts[t['node']],2)) for t in ap.tasks[k]])
for s in range(maxs):
    a=ap.get_action(obs)
    obs,r,term,trunc,info=env.step(a)
    if s%25==0 or term:
        r0=obs.get_object_from_name('rover0'); r1=obs.get_object_from_name('rover1')
        print(s,"r0",[round(float(obs.get(r0,f)),2) for f in ['x','y','store_full','calibrated','at_home']],
              "r1",[round(float(obs.get(r1,f)),2) for f in ['x','y','store_full','calibrated','at_home']],
              "task0",ap.tasks[0][0]['kind'] if ap.tasks[0] else None,
              "task1",ap.tasks[1][0]['kind'] if ap.tasks[1] else None, "term",term, flush=True)
    if term: break
for n in sorted(obs.get_object_names()):
    if n.startswith('objective') or n.startswith('sample'):
        o=obs.get_object_from_name(n)
        fs={f:round(float(obs.get(o,f)),2) for f in obs.type_features[o.type] if f not in ('x','y','z')}
        if any(v>0 for v in fs.values()): print(n,fs)
env.close()
