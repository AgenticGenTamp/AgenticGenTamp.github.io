import sys, numpy as np, pickle
from run_test import env
from approach import GeneratedApproach as G, in_goal, wrap
seed=int(sys.argv[1]); n=int(sys.argv[2]); T=int(sys.argv[3])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=G(env.action_space,env.observation_space,{}); ap.reset(s,info)
for t in range(T):
    s,*_=env.step(ap.get_action(s))
b,arm,g,cubes,_=ap._read(s)
print('wiper',ap.wiper.round(3))
for i,c in enumerate(cubes):
    if in_goal(c): continue
    ok=[round(th,2) for th in np.linspace(-np.pi,np.pi,73) if G._base_ok(ap._base_for(c,th)) and not G._arm_hits(ap._base_for(c,th)[:2],th)]
    print(i,c.round(3),'feasible',ok)
