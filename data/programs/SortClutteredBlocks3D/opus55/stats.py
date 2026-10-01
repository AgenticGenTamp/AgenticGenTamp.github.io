import sys, collections, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1]); cnt=int(sys.argv[2])
env=make_env(); obs,info=env.reset(seed=seed,options={"object_count":cnt})
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
steps=collections.Counter(); trans=collections.Counter(); last=None
for t in range(1000):
    a=ap.get_action(obs); steps[ap.phase]+=1
    if last and last!=ap.phase: trans[last+">"+ap.phase]+=1
    last=ap.phase
    obs,r,term,trunc,info=env.step(a)
    if term: break
ap._parse(obs)
print(seed, "inbin", sum(ap._in_bin(n) for n in ap.cube_names), dict(steps), {k:v for k,v in trans.items() if k in("lift>select","lift>putback","descend>select","transport>select","release>select","lift>transport")})
