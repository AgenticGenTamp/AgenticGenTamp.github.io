import numpy as np, time, cProfile, pstats
from env_client import make_env
from approach import GeneratedApproach
env=make_env(); obs,_=env.reset(seed=3)
ap=GeneratedApproach(); ap.reset(obs,{})
tA=0; tE=0
pr=cProfile.Profile()
for t in range(1000):
    t0=time.time(); pr.enable(); a=ap.get_action(obs); pr.disable(); t1=time.time()
    obs,r,te,tr,_=env.step(a); t2=time.time()
    tA+=t1-t0; tE+=t2-t1
    if te: break
print('steps',t+1,'approach time',tA,'env time',tE)
pstats.Stats(pr).sort_stats('cumulative').print_stats(8)
