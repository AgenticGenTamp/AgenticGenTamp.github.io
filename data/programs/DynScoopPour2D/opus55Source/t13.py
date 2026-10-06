import sys
from common import make_env
from approach import GeneratedApproach
seeds=[1,8,9,14,26,35,41,44,45,50,53,56]
for c in [1,2,5]:
    res=[]
    for sd in seeds:
        env=make_env(); s,info=env.reset(seed=sd,options={'object_count':c})
        ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
        for t in range(400):
            s,_,_,_,_=env.step(ap.get_action(s))
            if ap.phase not in ('start','approach','push','far_approach','grasp'): break
        res.append((ap.grasp_tries,t))
    print(c,res,flush=True)
