import cProfile, pstats, sys
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); obs,info=env.reset(seed=int(sys.argv[1]))
ap=GeneratedApproach(env.action_space, env.observation_space, None)
def run():
    global obs
    ap.reset(obs,info)
    for i in range(1000):
        obs,r,t,tr,_=env.step(ap.get_action(obs))
        if t or tr: break
    print('steps',i+1,t)
cProfile.run('run()','prof.out')
p=pstats.Stats('prof.out'); p.sort_stats('cumulative').print_stats(25)
