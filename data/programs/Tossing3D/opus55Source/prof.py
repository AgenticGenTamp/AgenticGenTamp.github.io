import cProfile, pstats, numpy as np
from common import make_env
from approach import GeneratedApproach
env=make_env([1])
appr=GeneratedApproach(env.action_space, env.observation_space, {})
s,info=env.reset(seed=0, options={'object_count':1}); appr.reset(s,info)
def run():
    global s
    for i in range(170):
        s,*_=env.step(appr.get_action(s))
cProfile.run('run()','prof.out')
pstats.Stats('prof.out').sort_stats('cumtime').print_stats(12)
