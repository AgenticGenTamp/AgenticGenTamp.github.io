import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
seed=int(sys.argv[1])
env=make_env(); ap=GeneratedApproach(env.action_space, env.observation_space, {})
obs,info=env.reset(seed=seed); ap.reset(obs,info)
print("path", [(round(a,3),round(b,3)) for a,b in ap.path])
print("radius", ap.radius, "bounds", ap._bounds)
prev=None
for i in range(120):
    a=ap.get_action(obs)
    obs,rew,term,tr,_=env.step(a)
    r=obs.get_object_from_name("robot")
    p=(round(float(obs.get(r,"x")),4),round(float(obs.get(r,"y")),4))
    if i<200: print(i,"idx",ap.idx,"act",np.round(a,3),"pos",p,"th",round(float(obs.get(r,"theta")),2))
    if term: print("TERM",i); break
env.close()
