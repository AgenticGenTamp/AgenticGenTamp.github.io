import numpy as np, sys
from test_approach import make, run
from approach import *
def lb(seed,count):
    env=make(); s,info=env.reset(seed=seed, options={'object_count':count})
    ap=GeneratedApproach(env.action_space, None, {}); ap.reset(s,info)
    r,rects=ap._parse(s); m=ap._robot_model(r); q0=ap._q(r)
    tgt=rects['target_block']; reg=rects['target_region']
    best=1e9
    chk=Checker(m,ap._obstacles(rects,{'target_block'}),[])
    for g in ap._grasp_cfgs(tgt,m):
        if not chk.free(g): continue
        c=ap._carry_estimate(m,g,tgt,reg)
        best=min(best,nsteps(q0,g)+c)
    return best
tot=[];tl=[]
for sd in range(int(sys.argv[2])):
    b=lb(sd,int(sys.argv[1])); ok,st,_,_=run(sd,int(sys.argv[1])); tot.append(st); tl.append(b)
    print(sd,b,st)
print(np.mean(tl),np.mean(tot))
