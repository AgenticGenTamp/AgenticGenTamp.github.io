import sys
from run_test import env
from approach import GeneratedApproach
seed=int(sys.argv[1]); n=int(sys.argv[2])
s,info=env.reset(seed=seed,options={'object_count':n})
ap=GeneratedApproach(env.action_space,env.observation_space,{}); ap.reset(s,info)
for t in range(env.max_steps_for_count(n)):
    s,r,term,tr,_=env.step(ap.get_action(s))
    if term: break
print(t, ' '.join('%s:%d'%x for x in ap.log))
b,arm,g,cubes,_=ap._read(s)
print('base',b.round(3),'wiper',ap.wiper.round(3))
for c in cubes: print(c.round(3))
