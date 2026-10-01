import sys, numpy as np
from env_client import make_env
from geo_approach_old import GeneratedApproach, _handle_offset
seed=int(sys.argv[1]); gx,gy=float(sys.argv[2]),float(sys.argv[3])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
nm='part1'; P=obs.get_object_from_name(nm)
ap.parts=[nm]; ap.goals[nm]=np.array([gx,gy])+_handle_offset(obs,P)
for i in range(60):
    st=ap.stage; a=ap.get_action(obs); obs,r,term,tr,_=env.step(a)
    print(i,st,a[10],np.round(ap._part_pose(obs,nm)[:3],4),ap._robot(obs)[2],term)
    if term: break
