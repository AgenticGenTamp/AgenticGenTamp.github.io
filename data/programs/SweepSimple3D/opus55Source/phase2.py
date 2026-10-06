import sys, numpy as np
from run_test import env
from approach import GeneratedApproach
seed=int(sys.argv[1]); n=int(sys.argv[2]); thr=int(sys.argv[3]) if len(sys.argv)>3 else 40
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
L=0
for t in range(300+150*n):
    a=ap.get_action(s); s,r,term,tr,_=env.step(a)
    while L<len(ap.log):
        nm,k=ap.log[L]; L+=1
        if k>=thr and t>100:
            b,arm,g,cubes,_=ap._read(s)
            print(t,nm,k,'base',b.round(3),'wiper',ap.wiper.round(3), 'tgt', None if ap.queue[0].base is None else ap.queue[0].base.round(3))
    if term: break
print('end',t)
