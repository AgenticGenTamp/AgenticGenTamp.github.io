import sys, time, cProfile, pstats
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); obs,info=env.reset(seed=seed)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
pr=cProfile.Profile(); comp=0
for t in range(1000):
    t0=time.time(); pr.enable(); a=ap.get_action(obs); pr.disable(); comp+=time.time()-t0
    obs,r,te,tr,_=env.step(a)
    if te: break
print('steps',t+1,'comp',comp)
pstats.Stats(pr).sort_stats('cumulative').print_stats(12)
