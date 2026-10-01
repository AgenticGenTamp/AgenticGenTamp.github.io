import sys, numpy as np
from env_client import make_env
from approach import GeneratedApproach
s=int(sys.argv[1]); env=make_env(); obs,info=env.reset(seed=s)
ap=GeneratedApproach(env.action_space, env.observation_space, {}); ap.reset(obs,info)
lh=None; lp=None
for t in range(1000):
    a=ap.get_action(obs); obs,r,te,tr,info=env.step(a)
    ho=obs.get_object_from_name('hook'); ro=obs.get_object_from_name('robot')
    h=tuple(round(float(obs.get(ho,f)),4) for f in ['x','y','theta','held'])
    rr=tuple(round(float(obs.get(ro,f)),3) for f in ['x','y','theta','arm_joint','finger_gap'])
    if h!=lh or ap.phase!=lp: print(t,ap.phase,'h',h,'r',rr); lh=h; lp=ap.phase
    if te or tr: break
