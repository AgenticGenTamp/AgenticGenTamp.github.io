import time, sys
from run_test import env
from approach import GeneratedApproach
n=int(sys.argv[1])
s,info=env.reset(seed=0,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
ta=te=0
for t in range(int(sys.argv[2])):
    t0=time.time(); a=ap.get_action(s); t1=time.time(); s,*_=env.step(a); t2=time.time()
    ta+=t1-t0; te+=t2-t1
print('policy',ta,'env',te)
