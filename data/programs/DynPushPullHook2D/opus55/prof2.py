import sys,time,os,cProfile,pstats
from env_client import make_env
from approach import GeneratedApproach
e=make_env()
o,i=e.reset(seed=int(sys.argv[1]),options={'object_count':10})
ap=GeneratedApproach(e.action_space,e.observation_space,{}); ap.reset(o,i)
pr=cProfile.Profile()
for t in range(1000):
    pr.enable(); a=ap.get_action(o); pr.disable()
    o,r,term,tr,_=e.step(a)
    if term or tr: break
pstats.Stats(pr).sort_stats('cumulative').print_stats(18)
