import numpy as np, sys, collections
from env_client import make_env
from approach import GeneratedApproach
n=None if len(sys.argv)<2 else (None if sys.argv[1]=='d' else int(sys.argv[1]))
seeds=range(20)
env=make_env(); cnt=collections.Counter(); tot=0
for s in seeds:
    obs,info=env.reset(seed=s) if n is None else env.reset(seed=s,options={"object_count":n})
    ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
    for k in range(1000):
        a=ap.get_action(obs); cnt[ap.phase]+=1; tot+=1
        obs,r,term,tr,info=env.step(a)
        if term: break
print(tot/len(list(seeds)), sorted(cnt.items(), key=lambda x:-x[1]))
env.close()
