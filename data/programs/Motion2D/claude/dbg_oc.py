import sys, numpy as np, math
from env_client import make_env
from approach import GeneratedApproach
oc=int(sys.argv[1]); seed=int(sys.argv[2])
env=make_env(); ap=GeneratedApproach(env.action_space, env.observation_space,{})
obs,info=env.reset(seed=seed,options={"object_count":oc}); ap.reset(obs,info)
print("goal_th",round(ap.goal_th,3),"margin",ap.margin,"path",[(round(a,3),round(b,3)) for a,b in ap.path])
r=obs.get_object_from_name("robot")
last=None; stuckat=None
for i in range(1000):
    a=ap.get_action(obs); obs,rew,term,tr,_=env.step(a)
    p=(round(float(obs.get(r,"x")),4),round(float(obs.get(r,"y")),4),round(float(obs.get(r,"theta")),3))
    if i<400 and (i%1==0) and i>=int(sys.argv[3] if len(sys.argv)>3 else 0) and i<int(sys.argv[3] if len(sys.argv)>3 else 0)+25:
        print(i,"idx",ap.idx,"replans",ap._replans,"act",np.round(a,3),"pos",p)
    if term: print("TERM",i); break
else: print("no term, final",p)
