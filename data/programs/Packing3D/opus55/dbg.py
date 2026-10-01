import sys
import numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1])
env = make_env(); obs, info = env.reset(seed=seed)
ap = GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs, info)
for nm in ap.parts:
    P=obs.get_object_from_name(nm); print(nm, P.type, ap._part_pose(obs,nm)[:3], 'goal', ap.goals[nm], obs.get(P,'triangle_type') if 'Tri' in str(P.type) else '')
for t in range(int(sys.argv[2]) if len(sys.argv)>2 else 80):
    q0=ap._robot(obs)[1]; st=ap.stage; cur=ap.cur
    a=ap.get_action(obs); obs,r,term,tr,_=env.step(a)
    q1=ap._robot(obs)[1]
    rej = np.allclose(q0,q1) and np.any(a[3:10]!=0)
    print(t, cur, st, 'grip',a[10], 'REJ' if rej else '', 'ga',ap._robot(obs)[2], {nm: np.round(ap._part_pose(obs,nm)[:3],3).tolist() for nm in ap.parts} if st in('check_open','open','close','check_grasp') else '')
    if term: print('TERM'); break
