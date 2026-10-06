import sys, collections
from common import make_env
from approach import GeneratedApproach
env=make_env([1,2]); appr=GeneratedApproach(env.action_space, env.observation_space, {})
for seed in [0,1]:
    s,info=env.reset(seed=seed, options={'object_count':2}); appr.reset(s,info); cnt=collections.Counter()
    for i in range(1200):
        a=appr.get_action(s); cnt[getattr(appr,'phase','?')]+=1
        s,r,term,_,_=env.step(a)
        if term: break
    print(seed, i+1, dict(cnt))
