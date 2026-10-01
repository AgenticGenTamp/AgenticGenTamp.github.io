import numpy as np
from env_client import make_env
from approach import GeneratedApproach
env=make_env()
for s in range(0,20):
    obs,info=env.reset(seed=s)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    out=[]
    for nm in ap.parts:
        P=obs.get_object_from_name(nm); t=str(P.type)
        out.append((nm, 'T%d'%int(obs.get(P,'triangle_type')) if 'Tri' in t else 'C', np.round(ap._part_pose(obs,nm)[:3],3).tolist()))
    print(s,out)
