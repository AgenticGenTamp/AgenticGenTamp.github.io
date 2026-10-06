import sys, approach
from common import make_env
from approach import GeneratedApproach
seeds=[1,8,9,14,26,35,41,44,45,50,53,56]
for c in [1,5]:
    out=[]
    for sd in seeds:
        env=make_env(); s,info=env.reset(seed=sd,options={'object_count':c})
        ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
        ap._parse(s); x0=ap.hook['x']
        for t in range(400):
            s,_,_,_,_=env.step(ap.get_action(s))
            if ap.phase=='grasp': break
        ap._parse(s); out.append((round(x0,3),round(ap.hook['x'],4)))
    print(c,out)
