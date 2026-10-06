import sys, numpy as np
from run_test import env
from approach import GeneratedApproach
seed=int(sys.argv[1]); n=int(sys.argv[2])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
for t in range(int(sys.argv[3])):
    a=ap.get_action(s); q=ap.queue[0]
    s,r,term,tr,_=env.step(a)
    if t%int(sys.argv[4])==0:
        b,arm,g,cubes,_=ap._read(s)
        print(t,q.name,'base',b.round(3),'tgt',q.base.round(3),'wiper',ap.wiper.round(3))
