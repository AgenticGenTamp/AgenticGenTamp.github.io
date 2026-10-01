import sys, numpy as np
from env_client import make_env
from geo_approach_old import GeneratedApproach, _handle_offset
def run(seed, pose_goals):
    env=make_env(); obs,info=env.reset(seed=seed)
    ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(obs,info)
    for nm in ap.parts:
        P=obs.get_object_from_name(nm); t='C'
        if 'Tri' in str(P.type): t='T%d'%int(obs.get(P,'triangle_type'))
        ap.goals[nm]=np.array(pose_goals[t])+_handle_offset(obs,P)
    for i in range(400):
        obs,r,term,tr,_=env.step(ap.get_action(obs))
        if term or tr: break
    fin={nm:np.round(ap._part_pose(obs,nm)[:3],3).tolist() for nm in ap.parts}
    print(seed,'term',term,'steps',i,fin,flush=True)
G={'C':(0.3,-0.075),'T0':(0.3,0.06),'T1':(0.25,0.01)}
for s in [1,6,7,5,0,2,13]: run(s,G)
G1={'C':(0.3,-0.075),'T0':(0.3,0.0),'T1':(0.25,-0.05)}
for s in [1,9]: run(s,G1)
