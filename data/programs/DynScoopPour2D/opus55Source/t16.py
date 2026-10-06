import sys, approach
from common import make_env
from approach import GeneratedApproach
p=tuple(float(v) for v in sys.argv[1].split(','))
approach.FAR_PARAMS=[p[:6]]
if len(p)>6: approach.B2_EXTRA=p[6]
if len(p)>7: approach.SLIDE_V=p[7]
import os
seeds=[int(v) for v in os.environ.get("SEEDS","1,8,9,14,26,35,41,44,45,50,53,56").split(",")]
tot=0; n=0; out=[]
for c in [int(v) for v in os.environ.get("CNTS","1,5").split(",")]:
    for sd in seeds:
        env=make_env(); s,info=env.reset(seed=sd,options={'object_count':c})
        ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
        for t in range(400):
            s,_,_,_,_=env.step(ap.get_action(s))
            if ap.phase not in ('start','approach','push','far_approach','grasp') or ap.grasp_tries>0: break
        ok = ap.grasp_tries==0 and t<399
        tot+=ok; n+=1; out.append(t if ok else -1)
print(sys.argv[1], f"{tot}/{n}", out)
