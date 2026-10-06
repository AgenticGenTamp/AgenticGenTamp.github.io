import sys, numpy as np
from common import make_env, hook, rob
import approach
from approach import GeneratedApproach
seed,count,xv=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])
GeneratedApproach._choose_descent=lambda self: xv
env=make_env(); s,info=env.reset(seed=seed,options={'object_count':count})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
for t in range(int(sys.argv[4])):
    a=ap.get_action(s); s,r,term,_,_=env.step(a)
    H=hook(s)
    if (ap.phase in('sweep','lift') and t%4==0) or term:
        print(t,ap.phase,'H',round(H['x'],2),round(H['y'],2),'O',[(round(x,2),round(y,2)) for x,y,_ in ap.objs])
    if term: print('TERM'); break
