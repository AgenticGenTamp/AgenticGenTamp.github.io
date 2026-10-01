import sys, numpy as np, approach
from env_client import make_env
env=make_env(); ap=approach.GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=int(sys.argv[1]),options={'object_count':int(sys.argv[2])}); ap.reset(obs,info)
nm=sys.argv[3]; prev=None
for t in range(int(sys.argv[4])):
    a=ap.get_action(obs); obs,r,term,trunc,info=env.step(a)
    o=obs.get_object_from_name(nm); p=np.array([obs.get(o,f) for f in ('pose_x','pose_y','pose_z')])
    if prev is None or np.linalg.norm(p-prev)>1e-3: print(t, np.round(p,3), 'act', np.round(a[:3],2), 'grip', a[10])
    prev=p
    if term: break
