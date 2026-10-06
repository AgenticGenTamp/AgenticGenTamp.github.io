import sys, numpy as np
from common import make_env, hook, rob
from approach import GeneratedApproach
seed,count,N=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3])
env=make_env(); s,info=env.reset(seed=seed,options={'object_count':count})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
for t in range(N):
    a=ap.get_action(s); s,r,term,_,_=env.step(a)
    R=rob(s); H=hook(s)
    if t%int(sys.argv[4])==0 or term: print(t,ap.phase,np.round(a,3),{k:round(v,3) for k,v in R.items()},{k:round(v,3) for k,v in H.items()})
    if term: print('TERM'); break
